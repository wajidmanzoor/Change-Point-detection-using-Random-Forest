from load import LoadData
from methods import DetectionMethhod
from tqdm import tqdm
import os


DATASET_PATH = "../dataset/"
RESULT_PATH = "../results/"
minimal_relative_segment_length = 0.01
file_name = "wine.csv"

n_segments_list = [20, 80]
n_observations_list = [250,353,500,707,1000,1414,2000,2828,4000,5656,8000,11313,16000,22627,32000,45254,64000,90509,128000]


detector = DetectionMethhod()

#Dirichlet
pbar = tqdm(total=len(n_observations_list))
pbar.set_description("Running Dirichlet Simulation")
for seed, n_observations in enumerate(n_observations_list):
    for n_segments in n_segments_list:
        data_loader = LoadData(seed=seed)
        X,original_cpt = data_loader.generate_dirichlet(n_observations=n_observations,n_segments=n_segments)
        predicted_change_points, method_names, time_taken = detector.run_all_methods(X,minimal_relative_segment_length,ecp=False,mnwbs=False)
        with open(os.path.join(RESULT_PATH,"dirichlet_results_varying_segments.txt"),"a") as file_:
            for predicted_cpt,method_name, time_ in zip(predicted_change_points,method_names,time_taken):
                    rand_score = data_loader.calculate_rand_score(original_cpt,predicted_cpt)
                    file_.write(f"{seed+1},Dirichlet,{method_name},{rand_score},{time_},{n_segments},{n_observations}\n")
        pbar.uppdate(1)
pbar.close()
                    
#Wine
file_path = os.path.join(DATASET_PATH,file_name)
pbar = tqdm(total=len(n_observations_list))
pbar.set_description("Running Wine Simulation")
for seed, n_observations in enumerate(n_observations_list):
    for n_segments in n_segments_list:
        data_loader = LoadData(seed=seed)
        X,original_cpt = data_loader.get_data_from_csv_with_noise(file_path,n_observations=n_observations,n_segments=n_segments)
        predicted_change_points, method_names, time_taken = detector.run_all_methods(X,minimal_relative_segment_length,ecp=False,mnwbs=False)
        with open(os.path.join(RESULT_PATH,"Wine_with_noise_results_varying_segments.txt"),"a") as file_:
            for predicted_cpt,method_name, time_ in zip(predicted_change_points,method_names,time_taken):
                    rand_score = data_loader.calculate_rand_score(original_cpt,predicted_cpt)
                    file_.write(f"{seed+1},Wine,{method_name},{rand_score},{time_},{n_segments},{n_observations}\n")
        pbar.uppdate(1)
pbar.close()





    
