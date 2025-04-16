import qnmfits
import numpy as np
import scipy
import corner
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from pathlib import Path

notebook_dir = Path().resolve()
sys.path.append(str(notebook_dir.parent))

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from qnmfits.spatial_mapping_functions import * 
from bayes_qnm_GP_likelihood import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_config import PlotConfig

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"

class MethodPlots:

    config = PlotConfig()
    config.apply_style()

    def __init__(self, id, N_MAX=7, T=100, T0_REF=17, include_Mf=True, include_chif=True):
        self.id = id
        self.N_MAX = N_MAX
        self.T = T
        self.T0_REF = T0_REF
        self.include_Mf = include_Mf
        self.include_chif = include_chif

        self.sim_main = SXS_CCE(id, lev="Lev5", radius="R2")
        self.sim_lower = SXS_CCE(id, lev="Lev4", radius="R2")

        self._align_waveforms()

        self.qnm_list = [(2, 2, n, 1) for n in np.arange(0, N_MAX + 1)]
        self.spherical_modes = [(2, 2)]

        # Get the true values for the spin and mass (i.e. the bondi data from metadata file)

        self.chif_mag_ref = self.sim_main.chif_mag
        self.Mf_ref = self.sim_main.Mf

        self.T0s = np.linspace(-25, 100, 65)

        #self._initialize_results()

        colors = self.config.colors
        self.custom_colormap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        self.fundamental_color_WN = to_hex('#395470')
        self.fundamental_color_GP = to_hex('#395471')
        self.overtone_color_WN = to_hex('#65858c')
        self.overtone_color_GP = to_hex('#65858d')


    def _align_waveforms(self):
        """
        Align waveforms and interpolate them onto the same time grid.
        """
        time_shift = get_time_shift(self.sim_main, self.sim_lower)
        self.sim_lower.zero_time = -time_shift
        self.sim_lower.time_shift()

        new_times = np.arange(self.sim_main.times[0], self.sim_main.times[-1], 0.1)
        self.sim_main_interp = sim_interpolator(self.sim_main, new_times)
        self.sim_lower_interp = sim_interpolator(self.sim_lower, new_times)


    def _initialize_results(self):
        """
        Initialize arrays to store results for mismatches, amplitudes, and significance.
        """
        num_T0s = len(self.T0s)
        num_modes = len(self.qnm_list)

        self.unweighted_mismatches_LS = np.zeros(num_T0s)
        self.weighted_mismatches_LS = np.zeros(num_T0s)

        self.unweighted_mismatches_WN = np.zeros(num_T0s)
        self.weighted_mismatches_WN = np.zeros(num_T0s)

        self.unweighted_mismatches_GP = np.zeros(num_T0s)
        self.weighted_mismatches_GP = np.zeros(num_T0s)

        self.unweighted_mismatches_noise = np.zeros(num_T0s)
        self.weighted_mismatches_noise = np.zeros(num_T0s)

        self.amplitudes_LS = np.zeros((num_T0s, num_modes))

        self.amplitudes_WN = np.zeros((num_T0s, num_modes))
        self.amplitudes_GP = np.zeros((num_T0s, num_modes))

        self.amplitudes_WN_percentiles = {p: np.zeros((num_T0s, num_modes)) for p in [10, 25, 50, 75, 90]}
        self.amplitudes_GP_percentiles = {p: np.zeros((num_T0s, num_modes)) for p in [10, 25, 50, 75, 90]}

        self.significances_WN = np.zeros((num_T0s, num_modes))
        self.significances_GP = np.zeros((num_T0s, num_modes))


    def load_tuned_parameters(self):
        """
        Load tuned kernel parameters for GP and WN fits.
        """
        with open(data_dir / 'tuned_params.pkl', 'rb') as f:
            params = pickle.load(f)
        self.tuned_param_dict_main = params[self.id]

        with open(data_dir / 'param_dict_sim_lm_full.pkl', 'rb') as f:
            param_dict_sim_lm = pickle.load(f)

        tuning_hyperparams_s = [0.29245605468749936] # This value was determined in get_kernel_params_alt.ipynb
        hyperparam_rule_dict_s = {"sigma_max": "multiply"}
        self.tuned_param_dict_wn = {
            mode: get_new_params(param_dict_sim_lm[self.id][mode], tuning_hyperparams_s, hyperparam_rule_dict_s)
            for mode in param_dict_sim_lm[self.id]
        }


    def compute_mf_chif(self):
        """
        Compute mass and spin parameters for each t0 using least-squares minimization.
        """
        initial_params = (self.Mf_ref, self.chif_mag_ref)
        Mf_RANGE = (self.Mf_ref * 0.5, self.Mf_ref * 1.5)
        chif_mag_RANGE = (0.1, 0.99)
        bounds = (Mf_RANGE, chif_mag_RANGE)

        # Get T0 reference values first (i.e. the values at t0 = 17)

        result = minimize(
            self._mf_chif_mismatch,
            initial_params,
            args=(self.T0_REF, self.T, self.spherical_modes),
            method="Nelder-Mead",
            bounds=bounds,
        )

        self.Mf_t0 = result.x[0]
        self.chif_t0 = result.x[1] 

        self.Mfs_chifs = np.zeros((len(self.T0s), 2))

        for i, t0 in enumerate(self.T0s):
            args = (t0, self.T, self.spherical_modes)
            result = minimize(
                self._mf_chif_mismatch,
                initial_params,
                args=args,
                method="Nelder-Mead",
                bounds=bounds,
            )
            self.Mfs_chifs[i] = result.x
            initial_params = result.x


    def _mf_chif_mismatch(self, Mf_chif_mag_list, t0, T, spherical_modes):
        """
        Compute the mismatch for given mass and spin parameters.
        """
        Mf, chif_mag = Mf_chif_mag_list
        best_fit = qnmfits.multimode_ringdown_fit(
            self.sim_main.times, self.sim_main.h, self.qnm_list, Mf, chif_mag, t0, t0_method="closest", T=T, spherical_modes=spherical_modes
        )
        return best_fit["mismatch"]


    def compute_fits(self):
        """
        Compute GP, WN, and LS fits for each t0 and store results.
        """
        for i, t0 in enumerate(self.T0s):
            print(f"t0 = {t0}")

            Mf = self.Mfs_chifs[i, 0]
            chif_mag = self.Mfs_chifs[i, 1]

            fit_GP = qnm_BGP_fit(
                self.sim_main.times,
                self.sim_main.h,
                self.qnm_list,
                self.Mf_ref,
                self.chif_mag_ref,
                t0,
                self.tuned_param_dict_main,
                kernel_main,
                t0_method="geq",
                T=self.T,
                spherical_modes=self.spherical_modes,
                include_chif=self.include_chif,
                include_Mf=self.include_Mf,
            )

            fit_WN = qnm_BGP_fit(
                self.sim_main.times,
                self.sim_main.h,
                self.qnm_list,
                self.Mf_ref,
                self.chif_mag_ref,
                t0,
                self.tuned_param_dict_wn,
                kernel_s,
                t0_method="geq",
                T=self.T,
                spherical_modes=self.spherical_modes,
                include_chif=self.include_chif,
                include_Mf=self.include_Mf,
            )

            fit_LS = qnmfits.multimode_ringdown_fit(
                self.sim_main.times,
                self.sim_main.h,
                self.qnm_list,
                Mf,
                chif_mag,
                t0,
                T=self.T,
                spherical_modes=self.spherical_modes,
            )

            mm_mask = (self.sim_main_interp.times >= t0 - 1e-9) & (self.sim_main_interp.times < t0 + self.T - 1e-9)
            mm_times = self.sim_main_interp.times[mm_mask]
            main_data = {(2, 2): self.sim_main_interp.h[(2, 2)][mm_mask]}
            lower_data = {(2, 2): self.sim_lower_interp.h[(2, 2)][mm_mask]}

            # Store results (mismatches, amplitudes, significance, etc.)
            self._store_results(i, fit_GP, fit_WN, fit_LS, main_data, lower_data, t0)


    def _store_results(self, i, fit_GP, fit_WN, fit_LS, main_data, lower_data, t0):
        """
        Store results for mismatches, amplitudes, and significance for a given t0.
        """
        # Mismatches
        self.unweighted_mismatches_LS[i] = unweighted_mismatch(fit_LS["model"], fit_LS["data"])
        self.weighted_mismatches_LS[i] = weighted_mismatch(fit_LS["model"], fit_LS["data"], fit_GP["inv_noise_covariance"])

        self.unweighted_mismatches_WN[i] = unweighted_mismatch(fit_WN["model"], fit_WN["data"])
        self.weighted_mismatches_WN[i] = weighted_mismatch(fit_WN["model"], fit_WN["data"], fit_GP["inv_noise_covariance"])

        self.unweighted_mismatches_GP[i] = unweighted_mismatch(fit_GP["model"], fit_GP["data"])
        self.weighted_mismatches_GP[i] = weighted_mismatch(fit_GP["model"], fit_GP["data"], fit_GP["inv_noise_covariance"])

        self.unweighted_mismatches_noise[i] = unweighted_mismatch(main_data, lower_data)
        self.weighted_mismatches_noise[i] = weighted_mismatch(main_data, lower_data, fit_GP['inv_noise_covariance'])

        # Amplitudes
        self.amplitudes_LS[i, :] = np.abs(fit_LS["C"])
        self.amplitudes_WN[i, :] = fit_WN["mean_abs_amplitude"]
        self.amplitudes_GP[i, :] = fit_GP["mean_abs_amplitude"]

        for p in [10, 25, 50, 75, 90]:
            self.amplitudes_WN_percentiles[p][i, :] = fit_WN["abs_amplitude_percentiles"][p]
            self.amplitudes_GP_percentiles[p][i, :] = fit_GP["abs_amplitude_percentiles"][p]

        # Significance
        self.significances_WN[i, :] = get_significance_list(self.qnm_list, fit_WN["mean"], fit_WN["fisher_matrix"])
        self.significances_GP[i, :] = get_significance_list(self.qnm_list, fit_GP["mean"], fit_GP["fisher_matrix"])


    def get_t0_ref_fits(self):
        """
        Get the fits for the reference t0 using class variables.
        """

        ref_fit_LS = qnmfits.multimode_ringdown_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_t0,
            self.chif_t0,
            self.T0_REF,
            T=self.T,
            spherical_modes=self.spherical_modes,
        )

        ref_fit_WN = qnm_BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.T0_REF,
            self.tuned_param_dict_wn,
            kernel_s,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        ref_fit_GP = qnm_BGP_fit(
            self.sim_main.times,
            self.sim_main.h,
            self.qnm_list,
            self.Mf_ref,
            self.chif_mag_ref,
            self.T0_REF,
            self.tuned_param_dict_main,
            kernel_main,
            t0_method="geq",
            T=self.T,
            spherical_modes=self.spherical_modes,
            include_chif=self.include_chif,
            include_Mf=self.include_Mf,
        )

        self.ref_params = []
        for re_c, im_c in zip(np.real(ref_fit_LS["C"]), np.imag(ref_fit_LS["C"])):
            self.ref_params.append(re_c)
            self.ref_params.append(im_c)

        self.ref_samples_WN  = scipy.stats.multivariate_normal(
            ref_fit_WN['mean'], ref_fit_WN['covariance'], allow_singular=True
        ).rvs(size=10000)

        self.ref_samples_GP = scipy.stats.multivariate_normal(
            ref_fit_GP['mean'], ref_fit_GP['covariance'], allow_singular=True
        ).rvs(size=10000)

        num_amplitude_params = len(self.qnm_list) * 2

        samples_re_WN = self.ref_samples_WN[:, :num_amplitude_params:2]
        samples_im_WN = self.ref_samples_WN[:, 1:num_amplitude_params:2]
        self.samples_abs_WN = np.sqrt(samples_re_WN**2 + samples_im_WN**2)

        samples_re_GP = self.ref_samples_GP[:, :num_amplitude_params:2]  
        samples_im_GP = self.ref_samples_GP[:, 1:num_amplitude_params:2]  
        self.samples_abs_GP = np.sqrt(samples_re_GP**2 + samples_im_GP**2)  

        log_amplitudes_WN = np.log(self.samples_abs_WN)
        self.samples_weights_WN = np.exp(-np.sum(log_amplitudes_WN, axis=1))

        log_amplitudes_GP = np.log(self.samples_abs_GP)
        self.samples_weights_GP = np.exp(-np.sum(log_amplitudes_GP, axis=1))

        self.param_list = [qnm for qnm in self.qnm_list for _ in range(2)] + ["chif"] + ["Mf"]  


    def plot_mismatch(self, output_path="outputs/mismatch_plot.pdf", show=False):
        """
        Generate the mismatch plot and save it to the specified path.
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(self.config.fig_width, self.config.fig_height * 1.7),
            sharex=True, gridspec_kw={"hspace": 0}
        )

        # Plot weighted mismatches
        ax1.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax1.plot(self.T0s, self.weighted_mismatches_GP, label="GP", color="k")
        ax1.plot(self.T0s, self.weighted_mismatches_WN, label="WN", ls="--", color="k")
        ax1.fill_between(self.T0s, 0, self.weighted_mismatches_noise, color="grey", alpha=0.5)
        ax1.set_xlim(self.T0s[0], self.T0s[-1])
        ax1.set_ylabel(r"$\mathcal{M}^{22}_K$")
        ax1.set_yscale("log")
        ax1.legend(frameon=False, loc="upper right", labelspacing=0.1)

        # Plot unweighted mismatches
        ax2.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax2.plot(self.T0s, self.unweighted_mismatches_GP, label="GP", color="k")
        ax2.plot(self.T0s, self.unweighted_mismatches_WN, label="WN", ls="--", color="k")
        ax2.fill_between(self.T0s, 0, self.unweighted_mismatches_noise, color="grey", alpha=0.5)
        ax2.set_xlim(self.T0s[0], self.T0s[-1])
        ax2.set_xlabel("$t_0 \ [M]$")
        ax2.set_ylabel(r"$\mathcal{M}^{22}$")
        ax2.set_yscale("log")
        ax2.legend(frameon=False, loc="upper right", labelspacing=0.1)

        # Save and/or show the plot
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


    def plot_amplitude(self, output_path="outputs/amplitude_plot.pdf", show=False):
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))
        colors = self.custom_colormap(np.linspace(0, 1, len(self.qnm_list)))

        for i, qnm in enumerate(self.qnm_list):
            decay_time = qnmfits.qnm.omega_list([qnm], self.chif_mag_ref, self.Mf_ref)[0].imag
            closest_time_index = np.argmin(np.abs(self.T0s - 0))
            C_tau = np.exp(decay_time * (self.T0s[closest_time_index] - self.T0s))
            ax.plot(self.T0s, self.amplitudes_GP_percentiles[50][:, i] * C_tau, label=f"{qnm[2]}", color=colors[i])
            ax.plot(self.T0s, self.amplitudes_WN_percentiles[50][:, i] * C_tau, linestyle="--", color=colors[i])
            ax.fill_between(
                self.T0s,
                self.amplitudes_GP_percentiles[25][:, i] * C_tau,
                self.amplitudes_GP_percentiles[75][:, i] * C_tau,
                alpha=0.2,
                color=colors[i],
            )
            ax.fill_between(
                self.T0s,
                self.amplitudes_WN_percentiles[25][:, i] * C_tau,
                self.amplitudes_WN_percentiles[75][:, i] * C_tau,
                alpha=0.2,
                color=colors[i],
            )

        solid_line = Line2D([0], [0], color="black", linestyle="-")
        dashed_line = Line2D([0], [0], color="black", linestyle="--")
        color_legend = ax.legend(
            title="$n$", title_fontsize=8, ncol=1, frameon=False, loc="center right", bbox_to_anchor=(1.23, 0.5), fontsize=7
        )
        line_legend = ax.legend(
            [solid_line, dashed_line],
            ["GP", "WN"],
            frameon=False,
            loc="upper left",
            ncol=1,
            fontsize=7,
        )
        ax.add_artist(color_legend)

        ax.axvline(self.T0_REF, color="k", alpha=0.3, lw=1)
        ax.set_xlim(-10, 30)
        ax.set_ylim(1e-3, 1e5)
        ax.set_xlabel("$t_0 \ [M]$")
        ax.set_ylabel(r"$|\hat{C}_{\alpha}|$")
        ax.set_yscale("log")

        plt.tight_layout()
        plt.subplots_adjust(right=1)
        fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=[color_legend])
        if show:
            plt.show()
        plt.close(fig)


    def plot_significance(self, output_path="outputs/significance_plot.pdf", show=False):
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))
        colors = self.custom_colormap(np.linspace(0, 1, len(self.qnm_list)))

        for i, qnm in enumerate(self.qnm_list):
            ax.plot(self.T0s, self.significances_GP[:, i], label=f"{qnm[2]}", color=colors[i])
            ax.plot(self.T0s, self.significances_WN[:, i], linestyle="--", color=colors[i])

        solid_line = Line2D([0], [0], color="black", linestyle="-")
        dashed_line = Line2D([0], [0], color="black", linestyle="--")
        color_legend = ax.legend(
            title="$n$", title_fontsize=8, ncol=1, frameon=False, loc="center right", bbox_to_anchor=(1.25, 0.5), fontsize=7
        )
        line_legend = ax.legend(
            [solid_line, dashed_line],
            ["GP", "WN"],
            frameon=False,
            loc="lower left",
            ncol=1,
            fontsize=7,
            bbox_to_anchor=(0, -0.05),
        )
        ax.add_artist(color_legend)

        ax.axvline(self.T0_REF, color="k", alpha=0.3)
        ax.set_xlabel("$t_0 \ [M]$")
        ax.set_ylabel(r"$\mathcal{S}_{\alpha}$")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, self.T0s[-1])

        plt.tight_layout()
        plt.subplots_adjust(right=1)
        fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=[color_legend])
        if show:
            plt.show()
        plt.close(fig)


    def plot_fundamental_kde(self, output_path="outputs/fundamental_corner.pdf", show=False):

        parameter_choice = [(2, 2, 0, 1)]

        labels = [
            rf"$\mathrm{{Re}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$" if i % 2 == 0 else rf"$\mathrm{{Im}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$"
            for param in parameter_choice
            for i in range(2)
        ]

        abs_indices_fundamental = [i for i, param in enumerate(self.qnm_list) if param in parameter_choice]
        indices_fundamental = [i for i, param in enumerate(self.param_list) if param in parameter_choice]

        samples_fundamental_WN = self.ref_samples_WN[:, indices_fundamental]
        samples_fundamental_GP = self.ref_samples_GP[:, indices_fundamental]

        samples_abs_fundamental_WN = self.samples_abs_WN[:, abs_indices_fundamental]
        samples_abs_fundamental_GP = self.samples_abs_GP[:, abs_indices_fundamental]

        df_samples_fundamental_WN = pd.DataFrame(samples_fundamental_WN, columns=labels)
        df_samples_fundamental_GP = pd.DataFrame(samples_fundamental_GP, columns=labels)

        df_samples_fundamental_WN['Dataset'] = 'WN'
        df_samples_fundamental_GP['Dataset'] = 'GP'

        # Create the jointplot
        g = sns.jointplot(
            data=df_samples_fundamental_GP,
            x=labels[0],
            y=labels[1],
            hue="Dataset",
            kind="kde",
            palette=[self.fundamental_color_GP],
            marginal_kws={"fill": False},
            height=self.config.fig_width,
            levels=[0.1, 0.5]
        )

        sns.kdeplot(df_samples_fundamental_WN[labels[0]], ax=g.ax_marg_x, color=self.fundamental_color_WN, linestyle='--', fill=False)
        sns.kdeplot(y=df_samples_fundamental_WN[labels[1]], ax=g.ax_marg_y, color=self.fundamental_color_WN, linestyle='--', fill=False)
        sns.kdeplot(x=df_samples_fundamental_WN[labels[0]], y=df_samples_fundamental_WN[labels[1]], ax=g.ax_joint, color=self.fundamental_color_WN, fill=False, levels=[0.1, 0.5])

        g.ax_joint.legend_.remove()

        g.ax_joint.plot(self.ref_params[indices_fundamental[0]], self.ref_params[indices_fundamental[1]], "*", color="#fc5151", markersize=10)

        # Get dashed lines for the WN contours
        for collection in g.ax_joint.collections:
            color = None
            if collection.get_edgecolor().size:
                olor = to_hex(collection.get_edgecolor()[0])
            elif collection.get_facecolor().size:
                color = to_hex(collection.get_facecolor()[0])
            if color == self.fundamental_color_WN:
                collection.set_linestyle('--')  # Set linestyle to dashed for WN

        # Add inset plot
        ax_inset = inset_axes(
            g.ax_joint,
            width="50%",
            height="20%",
            loc="lower right",
            borderpad=1,
            bbox_to_anchor=(0, 0.03, 1, 1),
            bbox_transform=g.ax_joint.transAxes
        )

        df_samples_abs_fundamental_WN = pd.DataFrame({'Amplitude': samples_abs_fundamental_WN.flatten(), 'Dataset': 'WN'})
        df_samples_abs_fundamental_GP = pd.DataFrame({'Amplitude': samples_abs_fundamental_GP.flatten(), 'Dataset': 'GP'})

        df_samples_abs_fundamental_WN['Weight'] = self.samples_weights_WN
        df_samples_abs_fundamental_GP['Weight'] = self.samples_weights_GP

        sns.kdeplot(
            data=df_samples_abs_fundamental_GP,
            x='Amplitude',
            color=self.fundamental_color_GP,
            label="GP (Prior 1)",
            linewidth=1,
            ax=ax_inset
        )
        sns.kdeplot(
            data=df_samples_abs_fundamental_WN,
            x='Amplitude',
            color=self.fundamental_color_WN,
            linestyle='--',
            linewidth=1,
            label="WN (Prior 1)",
            ax=ax_inset
        )

        sns.kdeplot(
            data=df_samples_abs_fundamental_GP,
            x='Amplitude',
            color=self.fundamental_color_GP,
            label="GP (Prior 2)",
            linewidth=0.5,
            weights='Weight',
            ax=ax_inset
        )
        sns.kdeplot(
            data=df_samples_abs_fundamental_WN,
            x='Amplitude',
            color=self.fundamental_color_WN,
            label="WN (Prior 2)",
            linestyle='--',
            linewidth=0.5,
            weights='Weight',
            ax=ax_inset
        )

        ax_inset.set_title(r"$|C_{\alpha}|$", fontsize=8)
        #ax_inset.set_xlim(0.19, 0.23)
        #ax_inset.set_ylim(0.0, 300)
        ax_inset.set_ylabel("")
        ax_inset.set_xlabel("")
        ax_inset.set_yticklabels([])
        ax_inset.yaxis.set_ticks([])
        ax_inset.tick_params(axis='both', which='major', labelsize=6)

        line_styles_inset = [Line2D([0], [0], color=self.fundamental_color_GP, linewidth=1, label='Prior 1'),
                     Line2D([0], [0], color=self.fundamental_color_WN, linewidth=0.5, label='Prior 2')]

        ax_inset.legend(handles=line_styles_inset, loc='upper left', frameon=False, ncol=1, fontsize=4)

        #g.ax_joint.set_xlim(-0.175, -0.135)
        #g.ax_joint.set_ylim(0.01, 0.07)

        line_styles = [Line2D([0], [0], color=self.fundamental_color_WN, linestyle='-', label='GP'),
                       Line2D([0], [0], color=self.fundamental_color_WN, linestyle='--', label='WN')]

        g.figure.legend(handles=line_styles, loc='upper left', frameon=False, bbox_to_anchor=(0.22, 0.84), ncol=2, fontsize=7)

        g.figure.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(g.figure)


    def plot_overtone_kde(self, output_path="outputs/overtone_corner.pdf", show=False):

        parameter_choice = [(2, 2, 2, 1)]

        labels = [
            rf"$\mathrm{{Re}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$" if i % 2 == 0 else rf"$\mathrm{{Im}}(C_{{({param[0]},{param[1]},{param[2]},+)}})$"
            for param in parameter_choice
            for i in range(2)
        ]

        abs_indices_overtone = [i for i, param in enumerate(self.qnm_list) if param in parameter_choice]
        indices_overtone = [i for i, param in enumerate(self.param_list) if param in parameter_choice]

        samples_overtone_WN = self.ref_samples_WN[:, indices_overtone]
        samples_overtone_GP = self.ref_samples_GP[:, indices_overtone]

        samples_abs_overtone_WN = self.samples_abs_WN[:, abs_indices_overtone]
        samples_abs_overtone_GP = self.samples_abs_GP[:, abs_indices_overtone]

        df_samples_overtone_WN = pd.DataFrame(samples_overtone_WN, columns=labels)
        df_samples_overtone_GP = pd.DataFrame(samples_overtone_GP, columns=labels)

        df_samples_overtone_WN['Dataset'] = 'WN'
        df_samples_overtone_GP['Dataset'] = 'GP'

        # Create the jointplot
        g = sns.jointplot(
            data=df_samples_overtone_GP,
            x=labels[0],
            y=labels[1],
            hue="Dataset",
            kind="kde",
            palette=[self.overtone_color_GP],
            marginal_kws={"fill": False},
            height=self.config.fig_width,
            levels=[0.1, 0.5]
        )

        sns.kdeplot(df_samples_overtone_WN[labels[0]], ax=g.ax_marg_x, color=self.overtone_color_WN, linestyle='--', fill=False)
        sns.kdeplot(y=df_samples_overtone_WN[labels[1]], ax=g.ax_marg_y, color=self.overtone_color_WN, linestyle='--', fill=False)
        sns.kdeplot(x=df_samples_overtone_WN[labels[0]], y=df_samples_overtone_WN[labels[1]], ax=g.ax_joint, color=self.overtone_color_WN, fill=False, levels=[0.1, 0.5])

        g.ax_joint.legend_.remove()

        g.ax_joint.plot(self.ref_params[indices_overtone[0]], self.ref_params[indices_overtone[1]], "*", color="#fc5151", markersize=10)

        # Get dashed lines for the WN contours
        for collection in g.ax_joint.collections:
            color = None
            if collection.get_edgecolor().size:
                color = to_hex(collection.get_edgecolor()[0])
            elif collection.get_facecolor().size:
                color = to_hex(collection.get_facecolor()[0])
            if color == self.overtone_color_WN:
                collection.set_linestyle('--')  # Set linestyle to dashed for WN

        g.ax_joint.axvline(0, color="black", linestyle=":", linewidth=1)
        g.ax_joint.axhline(0, color="black", linestyle=":", linewidth=1)

        ax_inset = inset_axes(
            g.ax_joint,
            width="50%",
            height="20%",
            loc="lower right",
            borderpad=1,
            bbox_to_anchor=(0, 0.03, 1, 1),
            bbox_transform=g.ax_joint.transAxes
        )

        df_samples_abs_overtone_WN = pd.DataFrame({'Amplitude': np.hstack((samples_abs_overtone_WN.flatten(), -samples_abs_overtone_WN.flatten())), 'Dataset': 'WN'})
        df_samples_abs_overtone_GP = pd.DataFrame({'Amplitude': np.hstack((samples_abs_overtone_GP.flatten(), -samples_abs_overtone_GP.flatten())), 'Dataset': 'GP'})

        df_samples_abs_overtone_WN['Weight'] = np.hstack((self.samples_weights_WN, self.samples_weights_WN))
        df_samples_abs_overtone_GP['Weight'] = np.hstack((self.samples_weights_GP, self.samples_weights_GP))

        sns.kdeplot(
            data=df_samples_abs_overtone_GP,
            x='Amplitude',
            color=self.overtone_color_GP,
            linewidth=1,
            label="GP (Prior 1)",
            ax=ax_inset
        )
        sns.kdeplot(
            data=df_samples_abs_overtone_WN,
            x='Amplitude',
            color=self.overtone_color_WN,
            linestyle='--',
            linewidth=1,
            label="WN (Prior 1)",
            ax=ax_inset
        )

        sns.kdeplot(
            data=df_samples_abs_overtone_GP,
            x='Amplitude',
            color=self.overtone_color_GP,
            label="GP (Prior 2)",
            linewidth=0.5,
            weights="Weight",
            ax=ax_inset,
        )
        sns.kdeplot(
            data=df_samples_abs_overtone_WN,
            x='Amplitude',
            color=self.overtone_color_WN,
            label="WN (Prior 2)",
            linestyle='--',
            linewidth=0.5,
            weights="Weight",
            ax=ax_inset,
        )

        ax_inset.set_title(r"$|C_{\alpha}|$", fontsize=8)
        #ax_inset.set_xlim(0, 2)
        #ax_inset.set_ylim(0, 2.5)
        ax_inset.set_ylabel("")
        ax_inset.set_xlabel("")
        ax_inset.set_yticklabels([])
        ax_inset.yaxis.set_ticks([])
        ax_inset.tick_params(axis='both', which='major', labelsize=6)

        line_styles_inset = [Line2D([0], [0], color=self.overtone_color_GP, linewidth=1, label='Prior 1'),
                     Line2D([0], [0], color=self.overtone_color_WN, linewidth=0.5, label='Prior 2')]

        ax_inset.legend(handles=line_styles_inset, loc='upper right', frameon=False, ncol=1, fontsize=6)

        #g.ax_joint.set_ylim(-4, 2)

        line_styles = [Line2D([0], [0], color=self.overtone_color_GP, linestyle='-', label='GP'),
                   Line2D([0], [0], color=self.overtone_color_WN, linestyle='--', label='WN')]

        g.figure.legend(handles=line_styles, loc='upper left', frameon=False, bbox_to_anchor=(0.22, 0.84), ncol=1, fontsize=7)

        g.figure.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(g.figure)


    def plot_mass_spin_corner(self, output_path="outputs/mass_spin_corner.pdf", show=False):
        parameter_choice = ["chif", "Mf"]

        indices_Chif_M = [i for i, param in enumerate(self.param_list) if param in parameter_choice]
        labels_Chif_M = parameter_choice

        samples_Chif_M_WN = self.ref_samples_WN[:, indices_Chif_M]
        samples_Chif_M_GP = self.ref_samples_GP[:, indices_Chif_M]

        df_wn_Chif_M = pd.DataFrame(samples_Chif_M_WN, columns=labels_Chif_M)
        df_main_Chif_M = pd.DataFrame(samples_Chif_M_GP, columns=labels_Chif_M)

        df_wn_Chif_M['Dataset'] = 'WN'
        df_main_Chif_M['Dataset'] = 'GP'

        # Create the jointplot
        g = sns.jointplot(
            data=df_main_Chif_M,
            x="chif",
            y="Mf",
            hue="Dataset",
            kind="kde",
            palette=[self.fundamental_color_GP],
            marginal_kws={"fill": False},
            height=self.config.fig_width,
            levels=[0.1, 0.5],
        )

        sns.kdeplot(df_wn_Chif_M[labels_Chif_M[0]], ax=g.ax_marg_x, color=self.fundamental_color_WN, linestyle='--', fill=False)
        sns.kdeplot(y=df_wn_Chif_M[labels_Chif_M[1]], ax=g.ax_marg_y, color=self.fundamental_color_WN, linestyle='--', fill=False)
        sns.kdeplot(x=df_wn_Chif_M[labels_Chif_M[0]], y=df_wn_Chif_M[labels_Chif_M[1]], ax=g.ax_joint, color=self.fundamental_color_WN, fill=False, levels=[0.1, 0.5])

        # --- Adjust the central plot (ax_joint) KDEs ---
        for collection in g.ax_joint.collections:
            color = None
            if collection.get_edgecolor().size:
                color = to_hex(collection.get_edgecolor()[0])
            elif collection.get_facecolor().size:
                color = to_hex(collection.get_facecolor()[0])

            if color == self.fundamental_color_WN:
                collection.set_linestyle('--')  # Set linestyle to dashed for WN

        # Add vertical and horizontal dotted lines at the truth values
        g.ax_joint.plot(self.chif_t0, self.Mf_t0, "*", color="#fc5151", markersize=10)
        g.ax_joint.plot(self.chif_mag_ref, self.Mf_ref, "x", color="#fc5151", markersize=10)

        # Add legend for the truth values
        g.ax_joint.legend(loc="upper right", frameon=False)

        # Add labels
        g.set_axis_labels(r"$\chi_f$", r"$M_f$")

        g.ax_joint.set_xlim(0.66, 0.72)
        g.ax_joint.set_ylim(0.935, 0.975)

        line_styles = [Line2D([0], [0], color=self.fundamental_color_GP, linestyle='-', label='GP'),
                   Line2D([0], [0], color=self.fundamental_color_WN, linestyle='--', label='WN')]

        g.figure.legend(handles=line_styles, loc='upper left', frameon=False, bbox_to_anchor=(0.2, 0.83))

        g.figure.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(g.figure)


def main():
    # Initialize the MethodPlots instance
    method_plots = MethodPlots(id="0001", N_MAX=7, T=100, T0_REF=17, include_Mf=True, include_chif=True)

    # Perform necessary computations
    method_plots.load_tuned_parameters()
    method_plots.compute_mf_chif()
    #method_plots.compute_fits()
    method_plots.get_t0_ref_fits()

    # Generate plots
    #method_plots.plot_mismatch(output_path="outputs/mismatch.pdf", show=False)
    #method_plots.plot_amplitude(output_path="outputs/amplitude.pdf", show=False)
    #method_plots.plot_significance(output_path="outputs/significance.pdf", show=False)
    method_plots.plot_fundamental_kde(output_path="outputs/fundamental_kde.pdf", show=False)
    method_plots.plot_overtone_kde(output_path="outputs/overtone_kde.pdf", show=False)
    method_plots.plot_mass_spin_corner(output_path="outputs/mass_spin_corner.pdf", show=False)

def main_iter():
    for id, T0_ref in zip(['0001', '0002', '0003', '0004'], [17, 21, 23, 26]):
        for extra_qnm in [[], [(3,2,0,1)]]:
            method_plots = MethodPlots(id=id, N_MAX=7, T=100, T0_REF=T0_ref, include_Mf=True, include_chif=True)
            method_plots.qnm_list += extra_qnm
            method_plots._initialize_results()

            # Perform necessary computations
            method_plots.load_tuned_parameters()
            method_plots.compute_mf_chif()
            method_plots.compute_fits()
            method_plots.get_t0_ref_fits()

            # Generate plots
            method_plots.plot_mismatch(output_path=f"outputs/mismatch_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_amplitude(output_path=f"outputs/amplitude_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_significance(output_path=f"outputs/significance_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_fundamental_kde(output_path=f"outputs/fundamental_kde_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_overtone_kde(output_path=f"outputs/overtone_kde_{id}_{extra_qnm}.pdf", show=False)
            method_plots.plot_mass_spin_corner(output_path=f"outputs/mass_spin_corner_{id}_{extra_qnm}.pdf", show=False)


if __name__ == "__main__":
    main_iter()