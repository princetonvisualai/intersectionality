import numpy as np
import sys
sys.path.append('./import_files/groupfair')
sys.path.append('./import_files')
from algos import groupfair, regularizer
from groupfair_utils import calc_acc, calc_ind_viol, calc_eo_viol, calc_jelly_eo_viol, calc_jelly_acc

methods = [groupfair.Plugin]
params_list = [{'B':[50], 'nu':np.logspace(-3,-1,5), 'T':[10000], 'lr': [0.01], 'fairness': ['EO']}]

metrics_list = [[('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_eo_viol(p, xp, y)), ('jelly_ind_viol', lambda p,x,xp,y: calc_jelly_eo_viol(p, xp, y)), ('jelly_accuracy', lambda p,x,xp,y: calc_jelly_acc(p,y))],
                [('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_eo_viol(p, xp, y)), ('jelly_ind_viol', lambda p,x,xp,y: calc_jelly_eo_viol(p, xp, y)), ('jelly_accuracy', lambda p,x,xp,y: calc_jelly_acc(p,y))],
                [('accuracy', lambda p,x,xp,y: calc_acc(p,y)), ('ind_viol', lambda p,x,xp,y: calc_eo_viol(p, xp, y)), ('jelly_ind_viol', lambda p,x,xp,y: calc_jelly_eo_viol(p, xp, y)), ('jelly_accuracy', lambda p,x,xp,y: calc_jelly_acc(p,y))]
                ]
