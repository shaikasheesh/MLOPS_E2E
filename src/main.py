import hydra
from data_pre_processing import preprocess_data
from train_model import train_model
from eval_model import evaluate



@hydra.main(config_path='F:\\Machine_Learning_Ops\\mlops_1\\config',config_name='pre-processing',version_base="1.2")
def main(config):
    print('pre-processing started........')
    preprocess_data(config)
    print('pre-processing completed')
    print('Training started........')
    train_model(config)
    print('Training completed')
    print('Evaluation started........')
    evaluate(config)
    print('Eval completed & Experiment logged')


if __name__ == "__main__":
    main()
