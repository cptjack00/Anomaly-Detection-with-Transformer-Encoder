import json
import os

num_stacks=[6,5,4,3,2,1]
lr=[0.01, 0.005, 0.003, 0.002, 0.001]
data=["scada2", "scada1"]
d_ff = [128, 64, 32]
auto_dims = [(200, 100), (200, 50), (100, 50), (100, 25), (150, 50), (80, 40), (150, 100), (50, 25)]
batch_size = [128, 64, 32]
# auto_dims = [(120, 40), (100, 40)]
auto_dims.extend([(120, 60), (140, 70), ( 160, 80),  (180, 90)])


for d in data:
    for num in num_stacks:
        for l in lr:
            for ff in d_ff:
                for auto in auto_dims:
                    for b in batch_size:
                        filename = "{}_CEN_AUTO_{}stacks_{}dff_{}auto_{}trans_{}lr_{}batch".format(d.upper(), num, ff, auto[0], auto[1], l, b).replace(".", "_")
                        print(filename)
                        config_path = os.path.join("./configs/to_be_run", "{}.json".format(filename))
                        configs = {
                            "experiment": filename,
                            "data_dir": "{}".format(d),
                            "auto_dataset": "{}".format(d),
                            "trans_dataset": "new_scada1",
                            "num_stacks": num,
                            "d_model": 16,
                            "d_ff": ff,
                            "num_heads": 1,
                            "dropout": 0.1,
                            "autoencoder_dims": auto[0],
                            "l_win": auto[1],
                            "pre_mask": int(auto[1] / 5 * 2),
                            "post_mask": int(auto[1] / 5 * 3),
                            "batch_size": b,
                            "shuffle": 1,
                            "dataloader_num_workers": 4,
                            "auto_num_epoch": 10,
                            "trans_num_epoch": 15,
                            "load_dir": "default"
                        }
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(configs, f, indent=4)

