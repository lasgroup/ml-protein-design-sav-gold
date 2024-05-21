from mutedpy.experiments.streptavidin.active_learning_2.next_fit import update_model_and_save, estimate_std
from mutedpy.experiments.streptavidin.streptavidin_loader import *

if __name__ == "__main__":

    # load old data
    x,y,dts = load_first_round()
    # load new data
    x2, y2, dts2 = load_second_round()
    x3, y3, dts3 = load_third_round()

    sigma_std1 = estimate_std(x, y)
    sigma_std2 = estimate_std(x2, y2)
    sigma_std3 = estimate_std(x3, y3)

    print (sigma_std1)
    print (sigma_std2)

    #### update geometric
    data = [(x,y,sigma_std1),(x2,y2,sigma_std2),(x3,y3,sigma_std3)]
    old_dict_location = "../active_learning_2/Geometric/params/final_model_params_Jan_05.p"
    new_dict_location = "Geometric/params/final_model_params_Jan_05.p"
    update_model_and_save(old_dict_location,data,new_dict_location)

    #### update animo-acid
    data = [(x,y,sigma_std1),(x2,y2,sigma_std2),(x3,y3,sigma_std3)]
    old_dict_location = "../active_learning_2/AA_model/params/final_model_params.p"
    new_dict_location = "AA_model/params/final_model_params_Jan_05.p"
    update_model_and_save(old_dict_location, data, new_dict_location)
