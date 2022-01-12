from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./data/training_hr_images'],
                      test_folders=["./data/testing_lr_images"],
                      min_size=50,
                      output_folder='./')
