from utils.constants import *
import numpy as np


class ExperimentSettings:
    """
    Wrapper class for experiment settings
    """

    def __init__(self, experiment_name, activity_name, correctness,
                 random_seed, n_train_subjects, n_validation_subjects,
                 n_time_steps, step, n_features, features_name,
                 num_filters, kernel_size, activation, lr,
                 batch_size, epochs, downsample_rate, positive_class_weight):
        self.experiment_stats_file = EXPERIMENT_STATS
        self.experiment_name = experiment_name
        self.activity_name = activity_name
        self.correctness = correctness

        self.random_seed = random_seed

        self.n_train_subjects = n_train_subjects
        self.n_validation_subjects = n_validation_subjects

        self.n_time_steps = n_time_steps
        self.step = step
        self.n_features = n_features
        self.features_name = features_name

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.lr = lr

        self.batch_size = batch_size
        self.epochs = epochs
        self.downsample_rate = downsample_rate
        self.positive_class_weight = positive_class_weight
        self.losoxv_stats = None
        self.cm = None
        self.histories = None

    def set_losoxv(self, losoxv_stats):
        self.losoxv_stats = losoxv_stats

    def set_cm(self, cm):
        self.cm = cm

    def save_experiment_stats(self):
        with open(self.experiment_stats_file, 'a') as f:
            f.write("\n{},".format(self.experiment_name))
            f.write("{},".format(self.activity_name))
            f.write("{},".format(self.correctness))
            f.write("{},".format(self.random_seed))
            f.write("{},".format(self.n_train_subjects))
            f.write("{},".format(self.n_validation_subjects))
            f.write("{},".format(self.n_time_steps))
            f.write("{},".format(self.step))
            f.write("{},".format(self.n_features))
            f.write("{},".format(self.features_name))
            f.write("{},".format(self.num_filters))
            f.write("{},".format(self.kernel_size))
            f.write("{},".format(self.activation))
            f.write("{},".format(self.lr))
            f.write("{},".format(self.batch_size))
            f.write("{},".format(self.epochs))
            f.write("{},".format(self.downsample_rate))
            f.write("{},".format(self.positive_class_weight))

            # accuracy, loss - train
            mean_accuracy_train = self.losoxv_stats['train_acc'].mean()
            mean_loss_train = self.losoxv_stats['train_loss'].mean()

            std_accuracy_train = self.losoxv_stats['train_acc'].std()
            std_loss_train = self.losoxv_stats['train_loss'].std()

            f.write("{},".format(mean_accuracy_train))
            f.write("{},".format(mean_loss_train))
            f.write("{},".format(std_accuracy_train))
            f.write("{},".format(std_loss_train))

            # accuracy, loss - validation
            mean_accuracy_valid = self.losoxv_stats['valid_acc'].mean()
            mean_loss_valid = self.losoxv_stats['valid_loss'].mean()

            std_accuracy_valid = self.losoxv_stats['valid_acc'].std()
            std_loss_valid = self.losoxv_stats['valid_loss'].std()

            f.write("{},".format(mean_accuracy_valid))
            f.write("{},".format(mean_loss_valid))
            f.write("{},".format(std_accuracy_valid))
            f.write("{},".format(std_loss_valid))

            # accuracy, f1, loss - test
            mean_accuracy_test = self.losoxv_stats['test_acc'].mean()
            mean_loss_test = self.losoxv_stats['test_loss'].mean()
            mean_f1_test, std_f1_test = mean_std_f1(self.cm)

            std_accuracy_test = self.losoxv_stats['test_acc'].std()
            std_loss_test = self.losoxv_stats['test_loss'].std()

            f.write("{},".format(mean_accuracy_test))
            f.write("{},".format(mean_f1_test))
            f.write("{},".format(mean_loss_test))
            f.write("{},".format(std_accuracy_test))
            f.write("{},".format(std_f1_test))
            f.write("{}\n".format(std_loss_test))

            f.close()


def mean_std_f1(cms):
    """
    Given a dictionary of confusion matrices, returns the mean and standard deviation of the f1 score
    :param cms: Dictionary with key = subject_name, item = confusion matrix (expected to be 2x2)
    :return: f1_mean, f1_std
    """
    f1_m = []

    for subject, cm in cms.items():
        precision = cms[subject][0][0] / (cms[subject][0][0] + cms[subject][1][0])
        recall = cms[subject][0][0] / (cms[subject][0][0] + cms[subject][0][1])
        f1 = (2 * precision * recall) / (precision + recall)

        f1_m.append(f1)
    f1_m = np.array(f1_m)
    return f1_m.mean(), f1_m.std()