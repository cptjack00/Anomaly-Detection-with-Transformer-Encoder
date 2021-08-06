import argparse
import json
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param: json_file
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)
        config_file.close()
    return config_dict


def save_config(config):
    filename = config["result_dir"] + \
        "training_config_l_win_{}_auto_dims_{}.json".format(
            config["l_win"], config["autoencoder_dims"])
    config_to_save = json.dumps(config)
    f = open(filename, "w")
    f.write(config_to_save)
    f.close()


def process_config(json_file):
    config = get_config_from_json(json_file)

    # create directories to save experiment results and trained models
    if config["load_dir"] == "default":
        save_dir = "../experiments/{}/{}".format(
            config["experiment"], config["auto_dataset"])
    else:
        save_dir = config["load_dir"]
    config["summary_dir"] = os.path.join(save_dir, "summary/")
    config["result_dir"] = os.path.join(save_dir, "result/")
    config["checkpoint_dir"] = os.path.join(
        save_dir, "checkpoints/")
    return config


def create_dirs(*dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c", "--config",
        metavar="C",
        default="None",
        help="The Configuration file")
    # argparser.add_argument(
    #     "-n", "--num-client",
    #     default=1,
    #     help="The number of clients participating in Federated Learning")
    args = argparser.parse_args()
    return args
