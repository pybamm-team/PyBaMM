import pybamm
import pytest
import random
from tests import no_internet_connection

data_loader = pybamm.DataLoader()


@pytest.mark.skipif(
    no_internet_connection(),
    reason="Network not available to download files from registry",
)  # Skip if no internet
def test_fetch():
    # Fetch a file from the registry and check it in local cache folder for its presence
    data_loader = pybamm.DataLoader()
    random_file = random.choice(list(data_loader.files.keys()))
    file_path = data_loader.get_data(random_file)
    assert file_path.is_file()


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
def test_registry_checksum():
    assert data_loader.show_registry(checksum=True) == data_loader.files


@pytest.mark.skipif(
    no_internet_connection(),
    reason="Network not available to download files from registry",
)
def test_registry():
    assert data_loader.show_registry() == list(data_loader.files)
