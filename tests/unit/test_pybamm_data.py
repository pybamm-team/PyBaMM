import pybamm
import pytest
from tests import no_internet_connection

data_loader = pybamm.DataLoader()


@pytest.mark.skipif(
    no_internet_connection(),
    reason="Network not available to download files from registry",
)  # Skip if no internet
def test_fetch():
    data_loader = pybamm.DataLoader()
    test_file = next(iter(data_loader.files.keys()))
    return data_loader.get_data(test_file).is_file()


@pytest.mark.skipif(
    no_internet_connection(),
    reason="Network not available to download files from registry",
)
def test_fetch_fake():
    # Try to fetch a fake file not present in the registry
    with pytest.raises(ValueError):
        data_loader.get_data("NotAfile.json")


@pytest.mark.skipif(
    no_internet_connection(),
    reason="Network not available to download files from registry",
)
def test_registry():
    # Checking if the file names returned are equal to the ones in the registry
    return data_loader.show_registry() == list(data_loader.files)
