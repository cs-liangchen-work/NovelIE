import logging
import torch
import os


from transformers import CONFIG_NAME, WEIGHTS_NAME

logging.basicConfig(level=logging.INFO)


def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    # Load checkpoint
    chkpt = torch.load('./models/' + xlm_checkpoint_path, map_location="cpu")

    state_dict = chkpt

    two_levels_state_dict = {}
    for k, v in state_dict.items():
        print(k)
        if "pred_layer" in k:
            print('----')
            two_levels_state_dict[k] = v
        else:
            two_levels_state_dict['module.'+k] = v
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + xlm_checkpoint_path

    print("Save PyTorch model to {}".format(pytorch_weights_dump_path))
    torch.save(two_levels_state_dict, pytorch_weights_dump_path)



bus_file_name = os.listdir('./models')
for file_name in bus_file_name:
    convert_xlm_checkpoint_to_pytorch(str(file_name), './ms')
