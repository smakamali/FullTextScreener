import csv
import os

data = [
    ["paper_id", "pdf_filename", "title", "authors", "year"],
    [15, "ALECE An Attention-based Learned Cardinality Estimator for SPJ Queries on Dynamic Workloads.pdf", "ALECE: An Attention-based Learned Cardinality Estimator for SPJ Queries on Dynamic Workloads", "Pengfei Li, Wenqing Wei, Rong Zhu, Bolin Ding, Jingren Zhou, and Hua Lu", 2023],
    [19, "Balsa Learning a Query Optimizer Without Expert.pdf", "Balsa: Learning a Query Optimizer Without Expert Demonstrations", "Yang, Zongheng and Chiang, Wei-Lin and Luan, Sifei and Mittal, Gautam and Luo, Michael and Stoica, Ion", 2022],
    [20, "Bao Making Learned Query Optimization Practical.pdf", "Bao: Making Learned Query Optimization Practical", "Ryan Marcus, Parimarjan Negi, Hongzi Mao, Nesime Tatbul, Mohammad Alizadeh, and Tim Kraska", 2021],
    [21, "BayesCard Revitalizing Bayesian Networks for Cardinality Estimation.pdf", "BayesCard: Revitilizing Bayesian Frameworks for Cardinality Estimation", "Ziniu Wu, Amir Shaikhha, Rong Zhu, Kai Zeng, Yuxing Han, Jingren Zhou", 2020],
    [26, "CORDS Automatic Discovery of Correlations and Soft Functional Dependencies.pdf", "CORDS: Automatic discovery of correlations and soft functional dependencies", "Ilyas, I.F.; Markl, V.; Haas, P.; Brown, P.; Aboulnaga, A.", 2004],
    [27, "Cost-based or Learning-based A Hybrid Query Optimizer for Query Plan Selection.pdf", "Cost-Based or Learning-Based? A Hybrid Query Optimizer for Query Plan Selection", "Xiang Yu, Chengliang Chai, Guoliang Li, and Jiabin Liu", 2022],
    [28, "Deep Reinforcement Learning for Join Order Enumeration.pdf", "Deep Reinforcement Learning for Join Order Enumeration", "Marcus, Ryan and Papaemmanouil, Olga", 2018],
    [30, "DeepDB Learn from Data, not from Queries.pdf", "DeepDB: Learn from Data, not from Queries!", "Benjamin Hilprecht, Andreas Schmidt, Moritz Kulessa, Alejandro Molina, Kristian Kersting, Carsten Binnig", 2020],
    [38, "Fauce Fast and Accurate Deep Ensembles with Uncertainty for Cardinality Estimation.pdf", "Fauce: fast and accurate deep ensembles with uncertainty for cardinality estimation", "Liu, J.; Dong, W.; Zhou, Q.; Li, D.", 2021],
    [41, "Identifying Robust Plans through Plan Diagram Reduction.pdf", "Identifying robust plans through plan diagram reduction", "Harish, D.; Darera, P.N.; Haritsa, J.R.", 2008],
    [47, "Join Order Selection with Deep Reinforcement Learning Fundamentals, Techniques, and Challenges.pdf", "Join Order Selection with Deep Reinforcement Learning: Fundamentals, Techniques, and Challenges", "Zhengtong Yan, Valter Uotila, Jiaheng Lu", 2023],
    [49, "Kepler Robust Learning for Faster Parametric Query Optimization.pdf", "Kepler: Robust Learning for Parametric Query Optimization", "Lyric Doshi, Vincent Zhuang, Gaurav Jain, Ryan Marcus, Haoyu Huang, Deniz Altinbüken, Eugene Brevdo, Campbell Fraser", 2023],
    [52, "Learned Cardinalities Estimating Correlated Joins with Deep Learning.pdf", "Learned Cardinalities: Estimating Correlated Joins with Deep Learning", "Andreas Kipf and Thomas Kipf and Bernhard Radke and Viktor Leis and Peter Boncz and Alfons Kemper", 2019],
    [57, "LEON A New Framework for ML-Aided Query Optimization.pdf", "LEON: ANewFrameworkforML-AidedQueryOptimization", "Xu Chen, Haitian Chen, Zibo Liang, Shuncheng Liu, Jinghong Wang, Kai Zeng, Han Su, Kai Zheng", 2023],
    [59, "LOGER A Learned Optimizer towards Generating Efficient and Robust Query Execution Plans.pdf", "LOGER: A Learned Optimizer towards Generating Efficient and Robust Query Execution Plans", "Chen, T.; Chen, H.; Gao Gaojun@Pku.Edu.Cn, J.; Tu Tu.Yaofeng@Zte.Com.Cn, Y.", 2023],
    [65, "Neo A Learned Query Optimizer.pdf", "Neo: A Learned query optimizer", "Marcus, R.; Negi, P.; Mao, H.; Zhang, C.; Alizadeh, M.; Kraska, T.; Papaemmanouil, O.; Tatbul, N.", 2019],
    [75, "Plan Bouquets A Fragrant Approach to Robust Query Processing.pdf", "Plan bouquets: A fragrant approach to robust query processing", "Dutt, A.; Haritsa, J.R.", 2016],
    [76, "Plan Bouquets Query Processing without Selectivity Estimation.pdf", "Plan bouquets: Query processing without selectivity estimation", "Dutt, A.; Haritsa, J.R.", 2014],
    [95, "Robust Query Driven Cardinality Estimation under Changing Workloads.pdf", "Robust Query Driven Cardinality Estimation under Changing Workloads", "Negi, P.; Marcus, R.; Wu, Z.; Madden, S.; Kipf, A.; Kraska, T.; Tatbul, N.; Alizadeh, M.", 2023],
    [97, "Robust Query Processing.pdf", "Robust query processing", "Karthik, S.", 2016],
    [100, "Robustness Metrics for Relational Query Execution Plans.pdf", "Robustness metrics for relational query execution plans", "Wolf, F.; Brendle, M.; May, N.; Willems, P.R.; Sattler, K.-U.; Grossniklaus, M.", 2018],
    [102, "Roq Robust Query Optimization Based on a Risk-aware Learned Cost Model.pdf", "Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model", "Amin Kamali and Verena Kantere and Calisto Zuzarte and Vincent Corvinelli", 2024],
    [113, "Simple Adaptive Query Processing vs. Learned Query Optimizers.pdf", "Simple Adaptive Query Processing vs. Learned Query Optimizers", "Yunjia Zhang, Yannis Chronis, Jignesh M. Patel, Theodoros Rekatsinas", 2023],
    [114, "SkinnerDB Regret-Bounded Query Evaluation via Reinforcement Learning.pdf", "SkinnerDB: Regret-bounded query evaluation via reinforcement learning", "Trummer, I.; Moseley, S.; Maram, D.; Jo, S.; Antonakakis, J.", 2018],
    [117, "Smooth Scan robust access path selection without cardinality estimation.pdf", "Smooth Scan: robust access path selection without cardinality estimation", "Borovica-Gajic, R.; Idreos, S.; Ailamaki, A.; Zukowski, M.; Fraser, C.", 2018],
    [124, "Thorough Data Pruning for Join Query in Database System.pdf", "Thorough Data Pruning for Join Query in Database System", "Jintao, G.; Zhanhuai, L.; Jian, S.", 2023],
    [125, "Towards a Robust Query Optimizer A Principled and Practical Approach.pdf", "Towards a robust query optimizer: A principled and practical approach", "Babcock, B.; Chaudhuri, S.", 2005],
    [126, "Uncertainty-aware Cardinality Estimation by Neural Network Gaussian Process.pdf", "Uncertainty-aware Cardinality Estimation by Neural Network Gaussian Process", "Kangfei Zhao, Jeffrey Xu Yu, Zongyan He, Hao Zhang", 2021],
    [132, "Zero-Shot Cost Models for Out-of-the-box Learned Cost Prediction.pdf", "Zero-shot cost models for out-of-the-box learned cost prediction", "Benjamin Hilprecht, Carsten Binnig", 2022]
]

# Output folder
output_dir = "metadata_csvs"
os.makedirs(output_dir, exist_ok=True)

# Header row
header = data[0]

# Number of data rows (excluding header)
data_rows = data[1:]

# Divide the data 3 by 3
group_size = 3
num_files = (len(data_rows) + group_size - 1) // group_size  # ceiling division

for i in range(num_files):
    filename = f"metadata_{i+1}.csv"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        start = i * group_size
        end = start + group_size
        writer.writerows(data_rows[start:end])

print(f"✅ Created {num_files} CSV files in '{output_dir}'")
