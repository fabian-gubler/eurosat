# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2-dev
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="kL-QvJGgYmaW" outputId="f8da27b4-d707-45da-e4b1-78ad9b67e116"
# pip install nemo-toolkit['all']

# + colab={"base_uri": "https://localhost:8080/"} id="zJQ4W3pobWuL" outputId="63bdaea0-9e12-4a1c-c6aa-696682a7edb0"
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(model_name='stt_de_conformer_ctc_large')

# + colab={"base_uri": "https://localhost:8080/"} id="MWh7778jcCLU" outputId="1737fe8c-3903-4313-c729-875b68a30f71"
model.cfg

# + id="nHyBlzJ9dGd1"
# !unzip "/content/send/.zip"

# + id="w61UOU5Zd6ZY"
train_manifest = "/content/send/manifest.json"

# + colab={"base_uri": "https://localhost:8080/"} id="xtg3pQtteAAr" outputId="c9b6d633-dcf3-4ddf-f950-bde1d6ea68bc"
print(a)

# + id="YyIydhzoeWd9"
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import datetime
from pathlib import Path
import torch
from sbb_project import consts
import copy
from pytorch_lightning.loggers import WandbLogger  
import wandb
from ruamel.yaml import YAML
import functools
import copy

# + id="a0n9EjDofsA1"
import pytorch_lightning as pl

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["4289bfe55724439383081eea4b2466d7", "9052abd8982d46b3893d3d537600ff20", "e2bdfe93eba1483b8c5d8deecc155206", "86f61d8e7e3943b588c6c5e12c838a01", "bb680c3adc8a4b6bb0ed78849eda4604", "eb2c1cc3e1a945e2be2871c5099d7070", "e50018dc8e95461caeeb93060bb0b459", "94892cf5f1af4c659a2758da9ec0ee33", "65276faba5c44b8fbdacd323bd8d5fa0", "a5e70812415049ca9a11e284f4aa992c", "67c2e2e1b7274a7aa7a38cfc3c05db5e", "94917263067e49428e1c4f0615895bef", "190a2286a8e84c39a67cd743727e8898", "d3afdadb17df437eb1bfaa4f62b8d397", "cb36ea705dc2443e90ffb4b94cb93c9b", "e129619afae0438f8afe2738b5a6271b", "09bf6ada25414df5be370c8202ba7efb", "5c13e8e0dd524888a7d2ea7e181e8de1", "7c4bcf7249884fbd9f2af078bd04cc71", "9ccc8fa042f741f8a4ba6614309df587", "f5122c07fef04b7db3780fa79725fcc7", "803a773c6e774696899451f8c82fc845", "951033d819074232a3800b9c4671df06", "3f51ca6baff44328b74db0e763c82252", "fbd896cf15bd463384f9bd0359d90286", "cce97916228c4bbeb9b528502b7a2526", "c0058f0a8faf4badaad3d64913a4b4ed", "a206c021280f4415b1a0cec1fd8b9355", "24cd7b57e4a347e0b33bb259e663ab67", "c3f62b50386b4740a155a7a913b9bdeb", "760a6f699c3a401b99117b0a784ebace", "d0301309aedd48bcab1e4ea1050bd594", "11dd1e21c242422f9b5e77339636bf75", "9afd14c3031945149ecf4a220c5d1ebe", "84215b90b44e44edb24e7bb51e8fef8c", "0925a4ec0d7946bdb5996c1a33354117", "1715a6b20b934a6984bc2524fcbfc2cf", "0dc8ae49c1d943c486d8c5224997b4dd", "e733422ce9ae46e886f819d9b4298733", "14b185db4be44c7ebacdf23b7e6ef7c5", "2bb56fb5226849de8fbc48f1ea61505b", "0861a13abdae4177b4b0168a15369354", "84ccfb56b4664c74af22f64889db7511", "51d32e9e83044aa9966352a3d16c7667", "3b8ff109e74a49c08005f81cf9059fdb", "22c7614b245e4e70a415ffefbfe3493e", "854ef0361fe34eb58e558a68680a9802", "e7254586d995479984d9f79de074bf8a", "079d676f64ad49248256474100a71d64", "820f4a42edb6432cad7ef6f870cb46a1", "c2831ad9d4c445b0bb5bee448abff085", "05219fcda61c4a4b9110841a56d7f090", "a4490996de9f416e9849c6278281291b", "1b1b4e6c69b34723bd775cf22f22772c", "8ddd5f0c4c214308b5331ff0a9e0a10f", "e46d3940539f47558c35062f9b1c88f4", "a3f6926d4dcc45a6bbccb698809d9dd6", "cb7b11d2740e488a87fb7567e174f0de", "803b03e1fd0740748bce5dab7f7aa12e", "a9e71e4dde9549328e3303b1b4298517", "df856647f862464589823c881b93e1b7", "2977af73b6584715bcb0c6a2fd18f2a6", "c24baf5a8bf8432280071d81ec928d59", "45dc51a94ce245d78922132b7540bed4", "82414d66dc934714ac46f8f8dcd8e03b", "df181e8179ab46849a4f55957f48e0d6", "4ce592e1c24849f98a57749cb74eeab9", "1fe62d3ec47a4173b869d4acea6eff63", "05e4982539ea407abf31bbca4567c542", "11158b19c6ac43dbad5e024502285d21", "30ac9bb4da1a4c9db630981253a4131d", "42eb8f60605d4bd18283e1a374ef1679", "ceeffa1e6e1c45e0828e188d59bdec05", "2a8324364b5a4dda87c6b1f11681dd90", "0612f6077bf04bcd85f7a6708e9bd735", "5a1139342ba64d79bf3e64b6c226dc7e", "da1f10a9cadb4246b6063e43c5eee1a6", "12fd7c12872c4f96818032da74095445", "b2c0dfcdfabc437d964b66d1b5431159", "4e0907b42aff47728198fe049ebfb66b", "235856b6431d43289ade28f3c4ae6163", "5f69df9527ca458e8d35ad20f0c6ee50", "4a51b794379d4af481c8606a5b7bd1f1", "7aafedf05b4a431da3348cd3b2f2fb5c", "3e3d0170bbde4f4c9acaf5cdb81d2479", "90210eab654f494082185e0848997ea9", "c1af00bf14dd48c08b971687aa91fb6c", "c992f159db3e49b0b1d5eeb02833913c", "52bd54a7167f42a99b1a8646b7d42f15", "4b709b8fac7d4fb19f94b95b1d3549ba", "6ecfa607fc5241bebdead524d16bcdb4", "9026667db90b47e0b393e95084004bd8", "0ee6c694dc874aef80b13222685ffb3e", "801132f56118416492621ad7e32b117f", "24c9a528f35e42518433dbfcff8f3618", "802134efd35347beb7916a7461e7acfb", "e4bb5046f5f34167b34463901d25c9a2", "3f878da7eee741ffb186e93491954328", "f1a8285854664ccebce50744c6575af4", "3da49d541be843ff86c920c04acdee03", "5d377af73d4d4338ba5e2e309d245207", "74aace012229444298d3a774e8c86207", "92ac89ca2103473e9020c826ba0d336f", "2f4af30419bd4398add6f0b2e52e3e7b", "6329dafd0afb4efeb8b536e1c7aeab65", "4cc4f0f5fa784e10ad26ed65d3aaa5d8", "4b1a933491044660a7554cf4e63015b2", "06ebfc21e69b4598a02032019c3a8596", "eef735f7203b44c4b5c0214dad0d47df", "291fc3d23f194722be4bde42801fdf33", "eb342295c52a4ee3b42b347b53c1b765", "6481f55c05f4491dbdb4e3455f2f1dca", "2fee9b7c0773463aaec1ec6ae02830f7", "bd4ee1de4e9c4b22bc93e77031cfc0f8", "6c47ce20e8014ef0a16d153c8e6d51aa", "72c798c56cb645b0b23ebb87b397b133", "b9d81265b69e4363875bab6fb5153edb", "c78672184f254ecb968392a5e4f1b038", "1f783ae417754a0fa6e4aef1bed6c72a", "5a39722a883b428b882bef4f5aaabd3f", "7c6863a7f0464274b6a40aeb0449183b", "79913a8fee8947fbaef7968f98eb92a8", "9b8ecdd2454942ee85768dd6d491cbeb", "3ea9e235c70a4e02b070ef16f856a2c5", "e3d1acd6d7d645deb4492cf459212a5f", "ca9419c4ec434ef39bebf830432ab8b0", "4ff51b17484d4440aa6fe35c293d556e", "c5f789b09e2e4c24a42e73d5824a9f7c", "a1f598faa7154b4a8a028557c9ac7f7c", "04ff8458824449cc92de040287fd39fb", "2495854abe074ed18382dcad12147e77", "2a04ba91e9244c1784afc79b68c60594"]} id="jgfg7Uaze0cC" outputId="0f22fbf2-12e9-4030-96ca-a37bb394c469"
sweep_iteration()


# + id="XGV-1jMtcFJQ"
def sweep_iteration():
    
    # set up W&B logger
    # wandb.init()    # required to have access to `wandb.config`
    # wandb_logger = WandbLogger(log_model='all')  # log final model
        
    trainer = pl.Trainer(max_epochs=10)
    
    # setup model - note how we refer to sweep parameters with wandb.config
    model = nemo_asr.models.ASRModel.from_pretrained(model_name='stt_de_conformer_ctc_large')

    model.set_trainer(trainer)
    
    model.cfg.train_ds.is_tarred = False
    
    model.cfg.train_ds.manifest_filepath = train_manifest
    model.cfg.validation_ds.manifest_filepath = str(train_manifest)
    model.cfg.test_ds.manifest_filepath = str(train_manifest)
   
    
    model.cfg.train_ds.max_duration = 45
    model.cfg.train_ds.batch_size = 4
    model.cfg.validation_ds.batch_size = 4
    model.cfg.test_ds.batch_size = 4
    
    # model.cfg.optim.lr = wandb.config.lr
    # model.cfg.spec_augment.freq_width = wandb.config.freq_width
    # model.cfg.spec_augment.freq_masks = wandb.config.freq_masks
    # model.cfg.spec_augment.time_width = wandb.config.time_width
    # model.cfg.spec_augment.time_masks = wandb.config.time_masks

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_test_data(model.cfg.test_ds)
    model.setup_optimization(model.cfg.optim)

    # train
    trainer.fit(model)
