"""Separate six-component checkpoint smoke; exact prior panel and production baseline."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from methods import ComponentScorers,METHODS
from pilot_candle_v4_five_router_methods import main

if __name__=='__main__':
    main(scorer_type=ComponentScorers,methods=METHODS,
         output_name='candle_v4_six_component_smoke_20261111')
