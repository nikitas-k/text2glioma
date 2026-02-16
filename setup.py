import setuptools

setuptools.setup(
    name="text2glioma",
    version="0.2.0",
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml"],
        "text2glioma.preprocessing": ["atlas_masks/**/*.nii.gz"],
    },
)