import os
import pandas as pd
import numpy as np
from scipy import stats
from utils import helper_functions as helpers
from transformers import AutoTokenizer, OPTForCausalLM, LlamaTokenizer, LlamaForCausalLM
import json
import jsonlines
import argparse

import torch


def get_perplexity_list(df, m, t):
    """
    Gets perplexities of all sentences in a DataFrame based on given model
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            perplexity = helpers.perplexity_score(row['comments_processed'], m, t)
        except Exception as ex:
            print(ex.__repr__())
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_perplexity_list_test(df, m, t, dem):
    """
    Gets perplexities of all sentences in a DataFrame(contains 2 columns of contrasting sentences) based on given model
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments in 2 columns
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    List of sentence perplexities
    """
    perplexity_list = []
    for idx, row in df.iterrows():
        try:
            if dem == 'black':
                perplexity = helpers.perplexity_score(row['comments_1'], m, t)
            else:
                perplexity = helpers.perplexity_score(row['comments_2'], m, t)
        except Exception as ex:
            perplexity = 0
        perplexity_list.append(perplexity)
    return perplexity_list


def get_model_perplexity(df, m, t):
    """
    Finds model perplexity based on average model loss over all sentences
    Parameters
    ----------
    df : pd.DataFrame
    DataFrame with Reddit comments
    m : model
    Pre-trained language model
    t : tokenizer
    Pre-trained tokenizer for the given model

    Returns
    -------
    Model perplexity
    """
    model_perplexity = helpers.model_perplexity(df['comments_processed'], m, t)
    return model_perplexity


def find_anomalies(data):
    """
    Find outliers in a given data distribution
    Parameters
    ----------
    data : list
    List of sentence perplexities

    Returns
    -------
    List of outliers
    """
    anomalies = []

    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies


def read_json(file_path):
    """
    json file loading function
    """

    file_path = os.path.expanduser(file_path)
    file_extension = file_path.split('.')[-1]

    if file_extension=="jsonl":
        with open(file_path, 'r') as f:
            json_list = []
            for line in f:
                json_list.append(json.loads(line))
            return json_list
    else:
        with open(file_path, 'r') as f:
            return json.load(f)


def write_json(file_path, dict_obj):
    """
    writes `dict_obj` to json file
    """
    with open(file_path, 'w') as f:
        json.dump(dict_obj, f, indent="")


def write_jsonlines(file_path, list_obj):
    """
    writes `list_obj` to jsonlines file
    """
    with jsonlines.open(file_path, "w") as writer:
        writer.write_all(list_obj)


def main(model_path, reduce=False):
    """
    main function
    """

    # the path of input data samples
    data_path = str(os.path.join(os.getcwd(), "data"))

    if "ind_bias" in model_path.lower():   # trained models
        json_name = f"{'+'.join(model_path.split('ind_bias')[-1].split('/')[1:])}.json"
    else:   # base models
        json_name = f"{model_path.split('/')[-1]}.json"

    eval_category = {
        "religion1": ["jews", "christians"],
        "religion2": ["muslims", "christians"],
        "race": ["black", "white"],
        "orientation": ["lgbtq", "straight"],
        "gender": ["female", "male"]
    }

    input_file_suffix = '_biased_test_reduced'

    if "opt" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    elif "llama" in model_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        model = LlamaForCausalLM.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    result = {}
    for eval_cat, groups in eval_category.items():
        
        group_1, group_2 = groups
        metric = {}

        df_1 = pd.read_csv(data_path + '/' + eval_cat + '/' + 'reddit_comments_' + eval_cat + '_' + group_1 + input_file_suffix + '.csv')
        df_2 = pd.read_csv(data_path + '/' + eval_cat + '/' + 'reddit_comments_' + eval_cat + '_' + group_2 + input_file_suffix + '.csv')

        perplexity_1 = get_perplexity_list(df_1, model, tokenizer)
        perplexity_2 = get_perplexity_list(df_2, model, tokenizer)

        df_all = pd.concat([df_1, df_2], ignore_index=True)
        model_ppl = get_model_perplexity(df_all, model, tokenizer)
        metric["ave ppl"] = model_ppl

        df_1['perplexity'] = perplexity_1
        df_2['perplexity'] = perplexity_2

        unfiltered_mean_1 = np.mean(perplexity_1)
        unfiltered_std_1 = np.std(perplexity_1)
        unfiltered_mean_2 = np.mean(perplexity_2)
        unfiltered_std_2 = np.std(perplexity_2)

        outlier_1 = find_anomalies(np.array(perplexity_1))
        outlier_2 = find_anomalies(np.array(perplexity_2))

        for i, (p1, p2) in enumerate(zip(perplexity_1, perplexity_2)):
            if p1 in outlier_1 or p2 in outlier_2:
                df_1.drop(df_1.loc[df_1['perplexity']==p1].index, inplace=True)
                df_2.drop(df_2.loc[df_2['perplexity']==p2].index, inplace=True)

        if reduce:
            print("------------------------ reduce -------------------------------")

            filtered_mean_1 = np.mean(df_1['perplexity'])
            filtered_std_1 = np.std(df_1['perplexity'])
            filtered_mean_2 = np.mean(df_2['perplexity'])
            filtered_std_2 = np.std(df_2['perplexity'])

            print('Mean and std of filtered perplexities group_1 - Mean {}, Std {}'.format(filtered_mean_1, filtered_std_1))
            print('Mean and std of filtered perplexities group_2 - Mean {}, Std {}'.format(filtered_mean_2, filtered_std_2))
            
            metric[group_1] = {
                "ppl - mean": filtered_mean_1,
                "ppl - std": filtered_std_1
            }
            metric[group_2] = {
                "ppl - mean": filtered_mean_2,
                "ppl - std": filtered_std_2
            }

            t_unpaired, p_unpaired = stats.ttest_ind(df_1['perplexity'].to_list(), df_2['perplexity'].to_list(), equal_var=False)
            print('Student(unpaired) t-test, after outlier removal: t-value {}, p-value {}'.format(t_unpaired, p_unpaired))
            t_paired, p_paired = stats.ttest_rel(df_1['perplexity'].to_list(), df_2['perplexity'].to_list())
            print('Paired t-test, after outlier removal: t-value {}, p-value {}'.format(t_paired, p_paired))

            metric["t-value"] = {
                "paired": t_paired,
                "unpaired": t_unpaired
            }
            metric["p-value"] = {
                "paired": p_paired,
                "unpaired": p_unpaired
            }

        else:
            print('Mean and std of unfiltered perplexities group_1 - Mean {}, Std {}'.format(unfiltered_mean_1, unfiltered_std_1))
            print('Mean and std of unfiltered perplexities group_2 - Mean {}, Std {}'.format(unfiltered_mean_2, unfiltered_std_2))

            metric[group_1] = {
                "ppl - mean": unfiltered_mean_1,
                "ppl - std": unfiltered_std_1
            }
            metric[group_2] = {
                "ppl - mean": unfiltered_mean_2,
                "ppl - std": unfiltered_std_2
            }
            
            t_unpaired, p_unpaired = stats.ttest_ind(perplexity_1, perplexity_2)
            print('Student(unpaired) t-test, unfiltered: t-value {}, p-value {}'.format(t_unpaired, p_unpaired))
            t_paired, p_paired = stats.ttest_rel(perplexity_1, perplexity_2)
            print('Paired t-test, unfiltered: t-value {}, p-value {}'.format(t_paired, p_paired))

            metric["t-value"] = {
                "paired": t_paired,
                "unpaired": t_unpaired
            }
            metric["p-value"] = {
                "paired": p_paired,
                "unpaired": p_unpaired
            }

        result[eval_cat] = metric

    # write the evaluation result to json file
    write_json(file_path=os.path.join(os.getcwd(), "results", json_name), dict_obj=result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", dest="model_path", type=str, required=True)
    parser.add_argument("-reduce", action='store_const', const=True, default=False, help="True if remove outliers, False if not remove outliers")
    args = parser.parse_args()

    main(model_path=args.model_path, reduce=args.reduce)