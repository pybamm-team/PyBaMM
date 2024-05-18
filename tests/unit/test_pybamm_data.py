import pybamm
import pytest
import random
import requests

data_loader = pybamm.DataLoader()


def no_internet_connection():
    try:
        requests.get("https://github.com", timeout=5)
        return False
    except requests.ConnectionError:
        return True


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
