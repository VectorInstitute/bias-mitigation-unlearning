import os
import numpy as np
import pandas as pd
import altair as alt
import argparse

from measure_bias import read_json


def plot_bar_line(df, title_str, x_label, y_label_bar, x_label_rename, y_label_line="", color_range=[], y_domain=[], is_grouped=False, column_label="", column_sort=[]):
    """
    plot bar chart with line on dual axis
    """

    plot_title = alt.TitleParams(title_str, anchor='middle')

    # grouped bar chart
    if is_grouped:

        bar = alt.Chart(df, title=plot_title).mark_bar().encode(
            x=alt.X(x_label, axis=alt.Axis(title=None, labels=False, ticks=False), sort=column_sort),
            y=alt.Y(y_label_bar),
            color=alt.Color(x_label, sort=column_sort),
            column=alt.Column(
                column_label,
                header=alt.Header(title=None, labelOrient='bottom')
            )
        )

        chart = bar.properties(width=200, height=150)
    
    else:

        base = alt.Chart(df, title=plot_title).encode(
            x=alt.X(x_label, title=x_label_rename, axis=alt.Axis(labelAngle=0), sort=column_sort)
        )

        bar = base.mark_bar().encode(
            y=alt.Y(y_label_bar, axis={"orient": "left"}),
            color=alt.Color(x_label,
                            legend=None,
                            scale=alt.Scale(
                                domain=column_sort,
                                range=color_range)))

        line = base.mark_line(point=True).encode(
            y=alt.Y(y_label_line, axis={"orient": "right"}, scale=alt.Scale(domain=y_domain)),
            color=alt.value("#d62728")
        )

        chart = (bar + line).resolve_scale(y="independent").properties(width=200, height=150)
  
    return chart


def main(method, model):
    """
    main visualization function
    """

    # load dpo dataframe that contains ppl results
    df_dpo = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_dpo_.csv"), dtype={"tv_sc": str})

    if method == "pcgu" and model == "llama2":

        # file name
        prefix = "pcgu+models+Llama-2-7b-hf+notall+inp+adv+0.0002+512+"
        suffix = "+model_3.json"

        # coefficients
        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "284787": "20",
            "355984": "25",
            "427181": "30",
            "498378": "35",
            "569575": "40"
        }

        # chart title
        title = "PCGU - Llama-2-7b"

        # perplexity dataframe metadata
        model_name = "Llama_2_7b_hf"
        coeff_col = "pcgu_k_perc"
        base_val = "0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_pcgu.csv"), dtype={coeff_col: str})

    elif method == "pcgu" and model == "opt6.7":

        prefix = "pcgu+models+opt-6.7b+notall+inp+adv+0.001+128+"
        suffix = "+model_3.json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "246457": "20",
            "308071": "25",
            "369686": "30",
            "431300": "35",
            "492914": "40"
        }

        title = "PCGU - OPT-6.7b"

        model_name = "opt_6.7b"
        coeff_col = "pcgu_k_perc"
        base_val = "0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_pcgu.csv"), dtype={coeff_col: str})

    elif method == "pcgu" and model == "opt2.7":

        prefix = "pcgu+models+opt-2.7b+notall+inp+adv+0.0004+256+"
        suffix = "+model_10.json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "157983": "20",
            "197479": "25",
            "236975": "30",
            "276471": "35",
            "315967": "40"
        }

        title = "PCGU - OPT-2.7b"

        model_name = "opt_2.7b"
        coeff_col = "pcgu_k_perc"
        base_val = "0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_pcgu.csv"), dtype={coeff_col: str})

    elif method == "pcgu" and model == "opt1.3":

        prefix = "pcgu+models+opt-1.3b+notall+inp+adv+0.0003+256+"
        suffix = "+model_5.json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "98985": "20",
            "123731": "25",
            "148478": "30",
            "173224": "35",
            "197970": "40"
        }

        title = "PCGU - OPT-1.3b"

        model_name = "opt_1.3b"
        coeff_col = "pcgu_k_perc"
        base_val = "0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_pcgu.csv"), dtype={coeff_col: str})

    elif method == "14k" and model == "opt1.3":

        prefix = "task_vector+stereo_14k+facebook+opt-1.3b-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (14k) - OPT-1.3b"

        model_name = "opt_1.3b"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_14k.csv"), dtype={coeff_col: str})

    elif method == "14k" and model == "opt2.7":

        prefix = "task_vector+stereo_14k+facebook+opt-2.7b-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (14k) - OPT-2.7b"

        model_name = "opt_2.7b"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_14k.csv"), dtype={coeff_col: str})

    elif method == "14k" and model == "opt6.7":

        prefix = "task_vector+stereo_14k+facebook+opt-6.7b-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (14k) - OPT-6.7b"

        model_name = "opt_6.7b"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_14k.csv"), dtype={coeff_col: str})

    elif method == "14k" and model == "llama2":

        prefix = "task_vector+stereo_14k+model-weights+Llama-2-7b-hf-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (14k) - Llama-2-7b"

        model_name = "Llama_2_7b_hf"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_14k.csv"), dtype={coeff_col: str})

    elif method == "2k" and model == "opt1.3":

        prefix = "task_vector+stereo_dpo+facebook+opt-1.3b-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (2k) - OPT-1.3b"

        model_name = "opt_1.3b"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_2k.csv"), dtype={coeff_col: str})

    elif method == "2k" and model == "opt2.7":

        prefix = "task_vector+stereo_dpo+facebook+opt-2.7b-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (2k) - OPT-2.7b"

        model_name = "opt_2.7b"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_2k.csv"), dtype={coeff_col: str})

    elif method == "2k" and model == "opt6.7":

        prefix = "task_vector+stereo_dpo+facebook+opt-6.7b-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (2k) - OPT-6.7b"

        model_name = "opt_6.7b"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_2k.csv"), dtype={coeff_col: str})

    elif method == "2k" and model == "llama2":

        prefix = "task_vector+stereo_dpo+model-weights+Llama-2-7b-hf-lora_adapter-"
        suffix = ".json"

        coeff = {
            "base" : "base",
            "dpo": "dpo",
            "0.2": "0.2",
            "0.4": "0.4",
            "0.6": "0.6",
            "0.8": "0.8",
            "1.0": "1.0"
        }

        title = "Task Vector (2k) - Llama-2-7b"

        model_name = "Llama_2_7b_hf"
        coeff_col = "tv_sc"
        base_val = "0.0"
        df_all = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "all_results_tv_2k.csv"), dtype={coeff_col: str})
    
    df_t_ppl = pd.DataFrame(columns=["variant", "t-value", "perplexity", "color"])

    idx = 0
    column_sort = []
    for coeff_key, coeff_value in coeff.items():

        column_sort.append(coeff_value)

        if coeff_key == "base":

            if model == "opt1.3":
                json_name_base = "opt-1.3b.json"
            elif model == "opt2.7":
                json_name_base = "opt-2.7b.json"
            elif model == "opt6.7":
                json_name_base = "opt-6.7b.json"
            elif model == "llama2":
                json_name_base = "Llama-2-7b-hf.json"
            results = read_json(os.path.join((os.getcwd()), "results", json_name_base))

            # perplexity
            df_t_ppl.loc[idx, "perplexity"] = df_all.loc[(df_all["model"]==model_name) & (df_all[coeff_col]==base_val), "wikitext_perplexity"].values[0]

            # assign color for bar chart
            df_t_ppl.loc[idx, "color"] = "#c7c7c7"

        elif coeff_key == "dpo":

            if model == "opt1.3":
                json_name = "dpo+facebook+opt-1.3B.json"
            elif model == "opt2.7":
                json_name = "dpo+facebook+opt-2.7b.json"
            elif model == "opt6.7":
                json_name = "dpo+facebook+opt-6.7B.json"
            elif model == "llama2":
                json_name = "dpo+model-weights+Llama-2-7b-hf.json"
            results = read_json(os.path.join(os.getcwd(), "results", json_name))

            # perplexity
            df_t_ppl.loc[idx, "perplexity"] = df_dpo.loc[(df_dpo["model"]==model_name) & (df_dpo["tv_sc"]=="0"), "wikitext_perplexity"].values[0]

            # assign color for bar chart
            df_t_ppl.loc[idx, "color"] = "#ffbb78"

        else:

            json_name = prefix + coeff_key + suffix
            results = read_json(os.path.join(os.getcwd(), "results", json_name))

            # perplexity
            df_t_ppl.loc[idx, "perplexity"] = df_all.loc[(df_all["model"]==model_name) & (df_all[coeff_col]==coeff_value), "wikitext_perplexity"].values[0]

            # assign color for bar chart
            df_t_ppl.loc[idx, "color"] = "#aec7e8"

        df_t_ppl.loc[idx, "variant"] = coeff_value

        t_ave = (abs(results["religion1"]["t-value"]["paired"]) + 
                abs(results["religion2"]["t-value"]["paired"]) + 
                abs(results["race"]["t-value"]["paired"]) + 
                abs(results["orientation"]["t-value"]["paired"]) + 
                abs(results["gender"]["t-value"]["paired"])) / 5
        df_t_ppl.loc[idx, "t-value"] = t_ave
        
        idx += 1
    
    df_t_ppl.to_csv(os.path.join(os.getcwd(), "results", "charts", "agg", f"{method}_{model}.csv"), index=False)
        
    # remove very large perplexity values
    df_t_ppl["perplexity"].where(df_t_ppl["perplexity"] <= 20, np.nan, inplace=True)

    # rescale the perplexity axis
    ppl_max = df_t_ppl["perplexity"].max()
    ppl_min = df_t_ppl["perplexity"].min()
    ppl_range = ppl_max - ppl_min
    domain_max = ppl_max + 0.1 * ppl_range
    domain_min = ppl_min - 0.1 * ppl_range

    if method == "pcgu":
        x_axis = "% weight vectors updated (k)"
    else:
        x_axis = "scaling coefficient"

    chart_bar_line = plot_bar_line(df=df_t_ppl,
                                   title_str=title,
                                   x_label="variant",
                                   y_label_bar="t-value",
                                   x_label_rename=x_axis,
                                   y_label_line="perplexity",
                                   color_range=df_t_ppl["color"].tolist(),
                                   y_domain=[domain_min, domain_max],
                                   column_sort=column_sort)
    png_name = method + "_" + model + ".png"
    chart_bar_line.save(os.path.join(os.getcwd(), "results", "charts", "agg", png_name), ppi=300)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, required=True, choices=["pcgu", "14k", "2k"])
    parser.add_argument("-model", type=str, required=True, choices=["opt1.3", "opt2.7", "opt6.7", "llama2"])
    args = parser.parse_args()

    main(method=args.method, model=args.model)