import pooch
import pathlib


class DataLoader:
    """
    Data Loader class for downloading and loading data files upstream at https://github.com/pybamm-team/pybamm-data/

    The following files are listed in the registry -

    COMSOL Results
    ---------------

    :footcite:t:`Andersson2019`
    :footcite:t:`Doyle1993`
    :footcite:t:`Harris2020`
    :footcite:t:`Marquis2019`
    :footcite:t:`Marquis2020`

    - comsol_01C.json
    - comsol_05C.json
    - comsol_1C.json
    - comsol_1plus1D_3C.json
    - comsol_2C.json
    - comsol_3C.json

    Kokam SLPB 75106100 discharge data from Ecker et al (2015)
    ----------------------------------------------------------

    :footcite:t:`Ecker2015i`
    :footcite:t:`Ecker2015ii`

    - Ecker_1C.csv
    - Ecker_5C.csv

    Enertech cells - discharge results for beginning of life
    --------------------------------------------------------

    :footcite:t:`Andersson2019`
    :footcite:t:`Doyle1993`
    :footcite:t:`Harris2020`
    :footcite:t:`Marquis2019`
    :footcite:t:`Ai2019`
    :footcite:t:`Deshpande2012`
    :footcite:t:`Timms2021`

    - 0.1C_discharge_U.txt
    - 0.1C_discharge_displacement.txt
    - 0.5C_discharge_T.txt
    - 0.5C_discharge_U.txt
    - 0.5C_discharge_displacement.txt
    - 1C_discharge_T.txt
    - 1C_discharge_U.txt
    - 1C_discharge_displacement.txt
    - 2C_discharge_T.txt
    - 2C_discharge_U.txt
    - 2C_discharge_displacement.txt
    - stn_2C.txt
    - stp_2C.txt


    Drive cycles
    ------------

    :footcite:t:`Andersson2019`
    :footcite:t:`Doyle1993`
    :footcite:t:`Harris2020`
    :footcite:t:`Marquis2019`
    :footcite:t:`Marquis2020`

    - UDDS.csv
    - US06.csv
    - WLTC.csv
    - car_current.csv


    """

    def __init__(self):
        """
        Create a pooch registry with the following data files available upstream at https://github.com/pybamm-team/pybamm-data/
        """
        self.version = "v1.0.0"  # Version of pybamm-data release
        self.path = pooch.os_cache("pybamm")
        self.files = {
            # COMSOL results
            "comsol_01C.json": "sha256:bc5136fe961e269453bdc31fcaa97376d6f8c347d570fd30ce4b7660c68ae22c",
            "comsol_05C.json": "sha256:3b044135ad88bdb88959304a33fe42b654d5ef7ef79d1271dd909cec55b257fb",
            "comsol_1C.json": "sha256:d45e3ab482c497c37ebbc68898da22bab0b0263992d8f2302502028bfd5ba0e9",
            "comsol_1plus1D_3C.json": "sha256:cdd5759202f9c7887d2ea6032f82212f2ca89297191fe5282b8812e1a09b1e1f",
            "comsol_2C.json": "sha256:15c2637f54bf1639621c58795db859cb08611c8182b7b20ade10e4c3e2839a5b",
            "comsol_3C.json": "sha256:11d5afccb70be85d4ac7e61d413c6e0f5e318e1635b1347c9a3c6784119711e6",
            # Kokam SLPB 75106100 discharge data from Ecker et al (2015)
            "Ecker_1C.csv": "sha256:428dc5113a6430492f430fb9e895f67d3e20f5643dc49a1cc0a922b92a5a8e01",
            "Ecker_5C.csv": "sha256:a89f8bf6e305b2a4195e1fae5e803277a40ed7557d263ef726f621803dcbb495",
            # Enertech cells - discharge results for beginning of life
            "0.1C_discharge_U.txt": "sha256:7b9fcd137441eea4ab686faee8d57fe242c5544400939ef358ccd99c63c9579d",
            "0.1C_discharge_displacement.txt": "sha256:f1329731ead5a82a2be9851cf80e4c6d68dd0774e07aee5361e2af3ab420d7be",
            "0.5C_discharge_T.txt": "sha256:2140b2f6bd698135d09a25b1f04c271d35a3a02999ace118b10389e01defa2ae",
            "0.5C_discharge_U.txt": "sha256:9ed8368b2c6149d2a69218e7df6aaade2511c9f7f6fc7932cda153d9a3a10f39",
            "0.5C_discharge_displacement.txt": "sha256:8098565ff99bc938864797b402f483c1c64a583d6db85d086f39ab0e7b638dd1",
            "1C_discharge_T.txt": "sha256:97308dfd7f7dd6c434e30f6c00fb6707c43c963855bb0800e0336809d5cc3756",
            "1C_discharge_U.txt": "sha256:8fc19de45172215d65c56522c224e6fc700ee443db236b814238a829b7a14c3a",
            "1C_discharge_displacement.txt": "sha256:c2e8617ac48a20921da1b40bbebac479a0a143edf16b12b2e1ff9aaaf1a32ff4",
            "2C_discharge_T.txt": "sha256:4bd688fb7653539701fe3df61857474b4d54e8b142c84fdc4c8b92b9573fa5d0",
            "2C_discharge_U.txt": "sha256:7b3c24b5e6df377075002abc2f62bab7c88b27d826812ba5a4c8385a1a12e723",
            "2C_discharge_displacement.txt": "sha256:2b11513d80827c762325c819a084b87b3a239af7d112f234c9871481760a0013",
            "stn_2C.txt": "sha256:bb2f90ccfd2cd86ad589287caae13470e554df2f4f47f0f583a5a7e3e6bd9d4c",
            "stp_2C.txt": "sha256:6fe73b3a18e5fcfb95151dfd7d34c3cbe929792631447ed3ec88c047c9778223",
            # Drive cycles
            "UDDS.csv": "sha256:9fe6558c17aad3cc08109186923aeb7459cd3097a381c44e854bf22dd12a5a2a",
            "US06.csv": "sha256:5909eb2ec7983fae86a050ff3b35a2041d0ab698710a6b0f95d5816e348077ba",
            "WLTC.csv": "sha256:bb2f95018a44ac1425cb9c787c34721192af502c7385f1358f28e4f75df11fd8",
            "car_current.csv": "sha256:4305b91b9df073cb048c25dd3fae725e06a94fe200e322e5c08db290d6799e36",
        }
        self.registry = pooch.create(
            path=self.path,
            base_url=f"https://github.com/pybamm-team/pybamm-data/releases/download/{self.version}",
            version=self.version,
            registry=self.files,
        )

    def get_data(self, filename: str):
        """
        Fetches the data file from upstream and stores it in the local cache directory under pybamm directory.

        Parameters
        ----------
        filename : str
            Name of the data file to be fetched from the registry.
        Returns
        -------
        pathlib.PurePath
        """
        self.registry.fetch(filename)
        return pathlib.Path(f"{self.path}/{self.version}/{filename}")

    def show_registry(self):
        """
        Prints the name of all the files present in the registry.

        Returns
        -------
        list
        """
        return list(self.files.keys())
