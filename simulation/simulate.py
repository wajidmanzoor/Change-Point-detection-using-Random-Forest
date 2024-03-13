from load import LoadData
from methods import DetectionMethhod
from tqdm import tqdm
import os


DATASET_PATH = "../dataset/"
RESULT_PATH = "../results/"
minimal_relative_segment_length = 0.01
file_paths = [os.path.join(DATASET_PATH,i) for i in os.listdir(DATASET_PATH)]

if not os.path.exists(RESULT_PATH):
  os.mkdir(RESULT_PATH)


detector = DetectionMethhod()
pbar = tqdm(total=100)
pbar.set_description("Running Simulation")
for seed in range(0,100):
    data_loader = LoadData(seed=seed)
    data,original_change_points,dataset_names = data_loader.get_all_data(file_paths)
    with open(os.path.join(RESULT_PATH,"simulation_results.txt"),"a") as file_:
        for X,orginal_cpt,dataset_name in zip(data,original_change_points,dataset_names):
            predicted_change_points, method_names, time_taken = detector.run_all_methods(X,minimal_relative_segment_length,ecp=False,mnwbs=False)
            for predicted_cpt,method_name, time_ in zip(predicted_change_points,method_names,time_taken):
                rand_score = data_loader.calculate_rand_score(orginal_cpt,predicted_cpt)
                file_.write(f"{seed+1},{dataset_name},{method_name},{rand_score},{time_},{str(orginal_cpt)},{str(predicted_cpt)}\n")
    pbar.update(1)
pbar.close()


    
