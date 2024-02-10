import hydra
import pandas as pd
from hydra import utils

#preprocessing steps
def encoding(df: pd.DataFrame):
    df_encoded = pd.get_dummies(df,drop_first= True)
    return df_encoded


@hydra.main(config_path='F:\\Machine_Learning_Ops\\mlops_1\\config',config_name='pre-processing')
def preprocess_data(config):
    cwd1 = utils.get_original_cwd() + '\\'
    emp_data = pd.read_csv(cwd1 + config.dataset.data,  encoding=config.dataset.encoding)
    emp_data_encoded = encoding(emp_data)
    emp_data_encoded.to_csv(cwd1 + config.processed.data_transformed)


if __name__ == '__main__':
    preprocess_data()

    


