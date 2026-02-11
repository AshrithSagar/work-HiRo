# tests/test_lasa.py

import pyLasaDataset as lasa  # type: ignore
from pacer.base import PACER
from pacer.lasa import LASADemonstrations


def test_lasa() -> None:
    demonstrations = LASADemonstrations(lasa.DataSet.GShape).to_demonstrations()
    pacer = PACER(demonstrations)
    pacer.prepare()
    pacer.train()


if __name__ == "__main__":
    test_lasa()
