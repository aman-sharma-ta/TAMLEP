import importlib


# Below function tests if the package installation was successful
def test_install():
    """function to test the package installation
    returns appropriate error message if the package was not installed successfully"""
    try:
        package_name = "house_pricing"
        importlib.import_module(package_name)
        installed = True
    except ImportError:
        installed = False

    assert (
        installed == True
    ), f"{package_name} - package was not imported.Try installing again!!"