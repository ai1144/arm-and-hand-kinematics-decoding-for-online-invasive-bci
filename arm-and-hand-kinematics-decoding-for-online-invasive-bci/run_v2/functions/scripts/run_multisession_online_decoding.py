"""Load only protocol-labeled attempt blocks"""
from dataclasses import replace
from pathlib import Path
import pandas as pd
from decoder_core.data import load_dataset

def combine_sessions(data_roots, template_dir, out_dir):
    datasets = [load_dataset(Path(root)) for root in data_roots]
    handles = sorted({int(t.handle_id) for ds in datasets for t in ds.trials})
    remap = {h:i for i,h in enumerate(handles)}
    trials=[]; rows=[]
    for ds in datasets:
        for t in ds.trials:
            if t.object_code != 'same_threshold_handle_to_geom':
                raise ValueError(f'Unconfirmed protocol mapping: {t.session} {t.object_code}')
            t=replace(t, gesture=remap[int(t.handle_id)])
            trials.append(t)
            rows.append(dict(session=t.session, gesture=t.gesture, handle_id=t.handle_id,
                             object_name=t.object_name, object_id=t.object_id,
                             object_label_source=t.object_code, repeat=t.repeat, n_time=t.n_time))
    combined=replace(datasets[0],trials=trials)
    table=pd.DataFrame(rows)
    table.to_csv(Path(out_dir)/'combined_trial_index.csv',index=False)
    return combined,table
