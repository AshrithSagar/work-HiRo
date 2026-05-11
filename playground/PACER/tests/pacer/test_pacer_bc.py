"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer.experiments import BCvsPACERBCExperiment

set_global_default_runtime_options(RuntimeOptions(validate=True))


if __name__ == "__main__":
    BCvsPACERBCExperiment(
        show_plots=True,
        demonstrations_choice="FROM_LASA",
        LASA_pattern="GShape",
        phase_estimator_choice="MLP",
        evaluate_phases=False,
        corruptions_choice=None,
        use_state_labels=False,
        filepath=None,
    ).run()
