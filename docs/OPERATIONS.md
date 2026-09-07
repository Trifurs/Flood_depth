# Operations

Use the model-named `configs/pa_hydrokan.xml` entry point for PA-HydroKAN. The
run directory is
self-contained and stores the resolved configuration, fingerprints, calibration
state, checkpoints, metrics, and environment metadata.

Before accepting a newly trained model, evaluate the raw checkpoint on the
validation split with the canonical output-validity mask. Keep the matching run
directory with the accepted checkpoint so its data and training identity can be
verified before deployment.
