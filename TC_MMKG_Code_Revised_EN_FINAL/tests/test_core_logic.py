from pathlib import Path
import pandas as pd

from scripts.common import jaccard
from scripts.rainflow2_use import extract_turning_points, rainflow_counting_strict
from scripts.try6_use import FatigueParams, equivalent_stress_range, cycles_to_failure


def test_jaccard():
    a=[('t','r','1'),('t','x','2')]
    b=[('t','r','1'),('t','y','3')]
    assert abs(jaccard(a,b)-1/3)<1e-12


def test_rainflow_nonempty():
    times=pd.Series(pd.date_range('2024-01-01',periods=7,freq='s'))
    vals=pd.Series([10,30,15,45,20,50,10])
    tp=extract_turning_points(times,vals)
    cycles=rainflow_counting_strict(tp)
    assert cycles
    assert all(c['range']>0 and c['count'] in (0.5,1.0) for c in cycles)


def test_fatigue_monotonicity():
    p=FatigueParams()
    low=equivalent_stress_range(100,-20,p)
    high=equivalent_stress_range(200,-20,p)
    assert high>low>0
    assert cycles_to_failure(high,p)<cycles_to_failure(low,p)
