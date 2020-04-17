import sys

sys.path.append('..')

from utils.constants import *
from utils.file_readers import *
from utils.data_exploration import *
from utils.signal_processing import *
from utils.sliding_window import *
from utils.stand_norm import *

import random
import pandas as pd

from sklearn import metrics
import seaborn as sns
import pickle

from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Conv1D, Dropout, MaxPooling1D, BatchNormalization
from keras import optimizers
from keras import regularizers


def comparative_plots_one_vs_all_activity(one_vs_all_activity=0):
    df_all = pd.read_csv('../Preprocessed/all_data.csv')

    # Create comparative plots across subjects for the signals to gauge how easily differentiable they are
    subjects = get_subject_names()
    correctness = 'correct'

    fig = plt.figure(figsize=(20, 40))
    fig.subplots_adjust(hspace=1, wspace=0.5)
    subject_counter = 0

    axes = ['accel_x_normalised', 'accel_y_normalised', 'accel_z_normalised']
    axes_counter = 0

    for i in range(1, 3 * len(subjects) + 1):
        subject = subjects[subject_counter]

        mask = (df_all['subject'] == subject) & (df_all['correctness'] == correctness) & (
                df_all['activity'] == one_vs_all_activity)

        df_filtered = df_all[mask]

        ax = fig.add_subplot(len(subjects), 3, i)
        plt.plot(df_filtered[axes[axes_counter]])
        plt.title("S_{}, {}".format(subject_counter, axes[axes_counter]))

        ax.set_ylim((0, 1))
        ax.set_xticklabels(range(len(df_filtered[axes[axes_counter]])))

        if i % 3 == 0:
            subject_counter += 1

        if axes_counter == 2:
            axes_counter = 1
        else:
            axes_counter += 1


def generate_range(n, end, start=0):
    return list(range(start, n)) + list(range(n + 1, end))


def initialise_model(num_filters=64, kernel_size=3, activation='relu',
                     n_time_steps=38, n_features=3, n_classes=10, architecture='shallow',
                     l1=0.01, l2=0.01, dropout_rate=0.2):
    if architecture == 'shallow':
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation, input_shape=(n_time_steps, n_features)))
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.summary()
        return model
    elif architecture == 'deep':
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation, input_shape=(n_time_steps, n_features)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.summary()
        return model
    
    elif architecture == 'batchnorm':
        model = Sequential()
        model.add(BatchNormalization(input_shape=(n_time_steps, n_features)))
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation, input_shape=(n_time_steps, n_features)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.summary()
        return model
    
    elif architecture=='dropout':
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation, input_shape=(n_time_steps, n_features)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters//2, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=num_filters//2, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv1D(filters=num_filters//4, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv1D(filters=num_filters//4, kernel_size=kernel_size,
                         activation=activation))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.summary()
        return model
    elif architecture == 'pooling':
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation, input_shape=(n_time_steps, n_features)))
        model.add(MaxPooling1D(pool_size=4, strides=2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation)) 
        model.add(BatchNormalization())
        
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(MaxPooling1D(pool_size=4, strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation=activation))
        model.add(MaxPooling1D(pool_size=4, strides=2)) 
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.summary()
        return model
    elif architecture == 'deep-regularisation':
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size,
                         activation='linear', input_shape=(n_time_steps, n_features),
                         activity_regularizer=regularizers.l1_l2(l1, l2)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        #
        # model.add(Conv1D(filters=num_filters // 2, kernel_size=kernel_size,
        #                  activation='linear', activity_regularizer=regularizers.l1_l2(l1, l2)))
        model.add(Conv1D(filters=num_filters // 2, kernel_size=kernel_size))
        model.add(Activation(activation))
        model.add(BatchNormalization())

        model.add(Conv1D(filters=num_filters // 2, kernel_size=kernel_size,
                         activation='linear', activity_regularizer=regularizers.l1_l2(l1, l2)))
        model.add(Activation(activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # model.add(Conv1D(filters=num_filters // 2, kernel_size=kernel_size,
        #                  activation='linear', activity_regularizer=regularizers.l1_l2(l1, l2)))
        model.add(Conv1D(filters=num_filters // 2, kernel_size=kernel_size))
        model.add(Activation(activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(filters=num_filters // 2, kernel_size=kernel_size,
                         activation='linear', activity_regularizer=regularizers.l1_l2(l1, l2)))
        model.add(Activation(activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.summary()
        return model




def losoxv_one_vs_all(experiment_name, one_vs_all_activity=0, random_seed=42, correctness='correct',
                      n_train_subjects=12, n_validation_subjects=2,
                      n_time_steps=38, step=19, n_features=3,
                      features=['accel_x_normalised', 'accel_y_normalised', 'accel_z_normalised'],
                      num_filters=64, kernel_size=3, activation='relu',
                      lr=0.0001, batch_size=32, epochs=200, downsample_rate=4, positive_class_weight=50.):
    # Loading data
    data = pd.read_csv("../Preprocessed/raw_data.csv")
    data = data.reindex(columns=['timestamp', 'seq', 'accel_x', 'accel_y', 'accel_z',
                                 'accel_magnitude', 'accel_pca',
                                 'subject', 'activity', 'correctness',
                                 'accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised',
                                 'accel_magnitude_standardised', 'accel_pca_standardised',
                                 'accel_x_normalised', 'accel_y_normalised', 'accel_z_normalised',
                                 'accel_magnitude_normalised', 'accel_pca_normalised'])

    random.seed(random_seed)

    n_classes = 2
    activities = 'all'

    subjects = get_subject_names()
    n_subjects = len(subjects)

    sgd = optimizers.SGD(lr=lr)

    # dictionaries for statistics
    # they will all have keys = left out subject and value = stats
    cm = {}
    histories = {}

    # dataframe for statistics
    losoxv_stats = pd.DataFrame(index=subjects, columns=['train_acc', 'train_loss',
                                                         'valid_acc', 'valid_loss',
                                                         'test_acc', 'test_loss'])

    save_path = "../Plots/{}/".format(experiment_name)

    for i in range(len(subjects)):

        #     early stopping for testing
        #     if i > 1:
        #         break

        # generate training, validation and test subjects

        left_out_subject = subjects[i]
        print("Left out subject = {}".format(subjects[i]))

        rng = generate_range(i, n_subjects)

        train_subjects = set()
        while (len(train_subjects) < n_train_subjects):
            choice = random.choice(rng)
            train_subjects.add(choice)

        # for validation, take the remaining ones
        valid_subjects = [subjects[j] for j in (set(rng) - train_subjects)]

        train_subjects = [subjects[j] for j in train_subjects]

        print("Test subjects = {}".format(train_subjects))
        print("Validation subjects = {}".format(valid_subjects))

        train_valid_subjects = train_subjects + valid_subjects

        # split the data
        mask_subj = (data['subject'] == train_valid_subjects[0])
        for tvs in train_valid_subjects:
            mask_subj = mask_subj | (data['subject'] == tvs)

        data_train = data[mask_subj]
        data_train.reset_index(drop=True, inplace=True)

        data_test = data[data['subject'] == left_out_subject]
        data_test.reset_index(drop=True, inplace=True)

        print("-" * 80)
        print("Removing outliers")
        print("-" * 80)

        # remove outliers for training
        data_train, _, _ = remove_outliers(data_train, activities=activities, subjects=train_valid_subjects,
                                           correctness=correctness)

        # remove outliers for test
        data_test, _, _ = remove_outliers(data_test, activities=activities, subjects=[left_out_subject],
                                          correctness=correctness)

        print("-" * 80)
        print("Standardising")
        print("-" * 80)

        # standardise for training
        data_train = standardise_data(data_train, activities=activities, subjects=train_valid_subjects,
                                      correctness=correctness)

        # standardise for test
        data_test = standardise_data(data_test, activities=activities, subjects=[left_out_subject],
                                     correctness=correctness)

        print("-" * 80)
        print("Normalising")
        print("-" * 80)

        # normalise for training
        data_train, scaler_fit = normalise_data(data_train, activities=activities, subjects=train_valid_subjects,
                                                correctness=correctness)

        # normalise for test
        data_test, _ = normalise_data(data_test, activities=activities, subjects=[left_out_subject],
                                      correctness=correctness, scaler_fit=scaler_fit)

        print("-" * 80)
        print("Generating datasets")
        print("-" * 80)

        # Generate datasets
        X_train, y_train = generate_dataset(df=data_train, n_time_steps=n_time_steps, n_features=n_features,
                                            step=step, features=features,
                                            one_vs_all_activity=one_vs_all_activity, subjects=train_subjects,
                                            correctness=correctness, downsample_rate=downsample_rate)

        print("Training set shapes: X_train = {}, y_train = {}".format(X_train.shape, y_train.shape))

        X_valid, y_valid = generate_dataset(df=data_train, n_time_steps=n_time_steps, n_features=n_features,
                                            step=step, features=features,
                                            one_vs_all_activity=one_vs_all_activity, subjects=valid_subjects,
                                            correctness=correctness, downsample_rate=downsample_rate)

        print("Valid set shapes: X_valid = {}, y_valid = {}".format(X_valid.shape, y_valid.shape))

        X_test, y_test = generate_dataset(df=data_test, n_time_steps=n_time_steps, n_features=n_features,
                                          step=step, features=features,
                                          one_vs_all_activity=one_vs_all_activity, subjects=[left_out_subject],
                                          correctness=correctness, downsample_rate=downsample_rate)

        print("Test set shapes: X_test = {}, y_test = {}".format(X_test.shape, y_test.shape))

        # create new model
        model = initialise_model(num_filters=num_filters, kernel_size=kernel_size, activation=activation,
                                 n_time_steps=n_time_steps, n_features=n_features, n_classes=n_classes)

        # compile model
        model.compile(optimizer=sgd,
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'mse'])

        class_weight = {0: positive_class_weight,
                        1: 1.}

        history = model.fit(X_train, y_train,
                            batch_size=batch_size, epochs=epochs,
                            validation_data=(X_valid, y_valid),
                            class_weight=class_weight)

        # stats
        y_pred_ohe = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred_ohe, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        activity_labels = get_activity_label_dict()
        cm_labels = [activity_labels[one_vs_all_activity], "Not {}".format(activity_labels[one_vs_all_activity])]

        confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
        fig = plt.figure(figsize=(10, 6))
        ax = sns.heatmap(confusion_matrix, xticklabels=cm_labels, yticklabels=cm_labels, annot=True, fmt='d')
        plt.xlabel("Predicted Labels", fontsize=14)
        plt.ylabel("True labels", fontsize=14)
        plt.title("CM - LOS = {}".format(left_out_subject))
        fig.tight_layout()
        plt.savefig(save_path + "{}_CM_LOS.pdf".format(left_out_subject))

        fig = plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title("Learning curve - LOS = {}".format(left_out_subject))
        plt.legend(fontsize=12)
        fig.tight_layout()
        plt.savefig(save_path + "{}_LC_LOS.pdf".format(left_out_subject))

        # record stats
        losoxv_stats.loc[left_out_subject, 'train_acc'] = history.history['accuracy'][-1]
        losoxv_stats.loc[left_out_subject, 'train_loss'] = history.history['loss'][-1]
        losoxv_stats.loc[left_out_subject, 'valid_acc'] = history.history['val_accuracy'][-1]
        losoxv_stats.loc[left_out_subject, 'valid_loss'] = history.history['val_loss'][-1]

        final_loss, final_acc, final_mse = model.evaluate(X_test, y_test, batch_size)
        print("Final loss = {}".format(final_loss))
        print("Final accuracy = {}".format(final_acc))

        losoxv_stats.loc[left_out_subject, 'test_acc'] = final_acc
        losoxv_stats.loc[left_out_subject, 'test_loss'] = final_loss

        # update cm and history
        cm[left_out_subject] = confusion_matrix
        histories[left_out_subject] = history

    # save everything
    losoxv_stats.to_csv("../Plots/{}/baseline_stats_losoxv.csv".format(experiment_name))
    pickle.dump(cm, open("../Plots/{}/cms.p".format(experiment_name), 'wb'))
    pickle.dump(histories, open("../Plots/{}/histories.p".format(experiment_name), 'wb'))

    # mean values of CM
    mean_cm = np.zeros(cm['adela'].shape)
    for subj in subjects:
        mean_cm += cm[subj]

    mean_cm /= len(subjects)

    fig = plt.figure(figsize=(10, 6))
    ax = sns.heatmap(mean_cm, xticklabels=cm_labels, yticklabels=cm_labels, annot=True)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True labels", fontsize=14)
    fig.tight_layout()
    plt.savefig("../Plots/{}/mean_cm_losoxv.pdf".format(experiment_name))

    # training and validation lines
    training_lines = []
    validation_lines = []

    for subj in subjects:
        training_lines.append(histories[subj].history['accuracy'])
        validation_lines.append(histories[subj].history['val_accuracy'])

    training_lines = np.asarray(training_lines)
    validation_lines = np.asarray(validation_lines)

    # mean and std of the learning curves
    training_mean = np.mean(training_lines, axis=0)
    validation_mean = np.mean(validation_lines, axis=0)

    training_std = np.std(training_lines, axis=0)
    validation_std = np.std(validation_lines, axis=0)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(training_mean, label='Training')
    plt.plot(validation_mean, '--', label='Validation')

    # Draw bands
    plt.fill_between(range(epochs), training_mean - training_std, training_mean + training_std, color="paleturquoise",
                     alpha=0.4)
    plt.fill_between(range(epochs), validation_mean - validation_std, validation_mean + validation_std,
                     color="moccasin", alpha=0.4)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim((0, 1))
    plt.legend(fontsize=14)

    fig.tight_layout()
    plt.savefig("../Plots/{}/mean_LC_losoxv.pdf".format(experiment_name))

    return losoxv_stats, cm, histories


def losoxv_all(experiment_name, random_seed=42, correctness='correct',
               n_train_subjects=12, n_validation_subjects=2,
               n_time_steps=38, step=19, n_features=3,
               features=['accel_x_normalised', 'accel_y_normalised', 'accel_z_normalised'],
               num_filters=64, kernel_size=3, activation='relu',
               lr=0.0001, batch_size=32, epochs=200, architecture='shallow', optimiser='sgd',
               remove_outliers_preprocess=True, normalisation='per_activity',
               l1=0.01, l2=0.01, dropout_rate=0.2):

    # Loading data
    data = pd.read_csv("../Preprocessed/raw_data.csv")
    data = data.reindex(columns=['timestamp', 'seq', 'accel_x', 'accel_y', 'accel_z',
                                 'accel_magnitude', 'accel_pca',
                                 'subject', 'activity', 'correctness',
                                 'accel_x_standardised', 'accel_y_standardised', 'accel_z_standardised',
                                 'accel_magnitude_standardised', 'accel_pca_standardised',
                                 'accel_x_normalised', 'accel_y_normalised', 'accel_z_normalised',
                                 'accel_magnitude_normalised', 'accel_pca_normalised'])

    random.seed(random_seed)

    n_classes = 10
    activities = 'all'

    subjects = get_subject_names()
    n_subjects = len(subjects)

    sgd = optimizers.SGD(lr=lr)
    adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # dictionaries for statistics
    # they will all have keys = left out subject and value = stats
    cm = {}
    histories = {}

    # dataframe for statistics
    losoxv_stats = pd.DataFrame(index=subjects, columns=['train_acc', 'train_loss',
                                                         'valid_acc', 'valid_loss',
                                                         'test_acc', 'test_loss'])

    save_path = "../Plots/{}/".format(experiment_name)

    for i in range(len(subjects)):

        #         if i > 1:
        #             break

        # generate training, validation and test subjects
        left_out_subject = subjects[i]
        print("Left out subject = {}".format(subjects[i]))

        rng = generate_range(i, n_subjects)

        train_subjects = set()
        while (len(train_subjects) < n_train_subjects):
            choice = random.choice(rng)
            train_subjects.add(choice)

        # for validation, take the remaining ones
        valid_subjects = [subjects[j] for j in (set(rng) - train_subjects)]

        train_subjects = [subjects[j] for j in train_subjects]

        print("Test subjects = {}".format(train_subjects))
        print("Validation subjects = {}".format(valid_subjects))

        train_valid_subjects = train_subjects + valid_subjects

        # split the data
        mask_subj = (data['subject'] == train_valid_subjects[0])
        for tvs in train_valid_subjects:
            mask_subj = mask_subj | (data['subject'] == tvs)

        data_train = data[mask_subj]
        data_train.reset_index(drop=True, inplace=True)

        data_test = data[data['subject'] == left_out_subject]
        data_test.reset_index(drop=True, inplace=True)

        if remove_outliers_preprocess:
            print("-" * 80)
            print("Removing outliers")
            print("-" * 80)

            # remove outliers for training
            data_train, _, _ = remove_outliers(data_train, activities=activities, subjects=train_valid_subjects,
                                               correctness=correctness)

            # remove outliers for test
            data_test, _, _ = remove_outliers(data_test, activities=activities, subjects=[left_out_subject],
                                              correctness=correctness)

        print("-" * 80)
        print("Standardising")
        print("-" * 80)

        # standardise for training
        data_train = standardise_data(data_train, activities=activities, subjects=train_valid_subjects,
                                      correctness=correctness)

        # standardise for test
        data_test = standardise_data(data_test, activities=activities, subjects=[left_out_subject],
                                     correctness=correctness)

        print("-" * 80)
        print("Normalising")
        print("-" * 80)

        if normalisation == 'per_activity':
            # normalise for training
            data_train, scaler_fit = normalise_data(data_train, activities=activities, subjects=train_valid_subjects,
                                                    correctness=correctness)

            # normalise for test
            data_test, _ = normalise_data(data_test, activities=activities, subjects=[left_out_subject],
                                          correctness=correctness, scaler_fit=scaler_fit)

        elif normalisation == 'universal':
            # normalise for training
            data_train, scaler_fit = normalise_data_universal(data_train, activities=activities, subjects=train_valid_subjects,
                                                    correctness=correctness)

            # normalise for test
            data_test, _ = normalise_data_universal(data_test, activities=activities, subjects=[left_out_subject],
                                          correctness=correctness, scaler_fit=scaler_fit)

        elif normalisation == 'cross':
            # normalise each activity with normalisers from all other activities
            # but keep the correctness separation
            data_train, scaler_fit = normalise_data(data_train, activities=activities, subjects=train_valid_subjects,
                                                    correctness=correctness)

            # normalise for test
            data_test, _ = normalise_data(data_test, activities=activities, subjects=[left_out_subject],
                                                    correctness=correctness, scaler_fit=scaler_fit)

            # Generate datasets
            X_train, y_train = generate_dataset(df=data_train, n_time_steps=n_time_steps, n_features=n_features,
                                                step=step, features=features,
                                                one_vs_all_activity='all', subjects=train_subjects,
                                                correctness=correctness)

            X_valid, y_valid = generate_dataset(df=data_train, n_time_steps=n_time_steps, n_features=n_features,
                                                step=step, features=features,
                                                one_vs_all_activity='all', subjects=valid_subjects,
                                                correctness=correctness)

            X_test, y_test = generate_dataset(df=data_test, n_time_steps=n_time_steps, n_features=n_features,
                                              step=step, features=features,
                                              one_vs_all_activity='all', subjects=[left_out_subject],
                                              correctness=correctness)

            print("Initial lengths of datasets = {}, {}, {}".format(len(X_train), len(X_valid), len(X_test)))
            print("Initial lengths of labels = {}, {}, {}".format(len(y_train), len(y_valid), len(y_test)))

            print("*"*80)
            print("Cross normalising...")
            print("*" * 80)
            for act in range(10):
                print("Activity {}".format(act))
                print("Cross normalising training")
                data_train_cross = cross_normalise_data(data_train, activity_norm=act, subjects=train_valid_subjects,
                                                        correctness=correctness, scaler_fit=scaler_fit)
                print("Cross normalising test")
                data_test_cross = cross_normalise_data(data_test, activity_norm=act, subjects=[left_out_subject],
                                                       correctness=correctness, scaler_fit=scaler_fit)

                print("Training samples")
                X_train_cross, y_train_cross = generate_dataset(df=data_train_cross, n_time_steps=n_time_steps, n_features=n_features,
                                                                step=step, features=features,
                                                                one_vs_all_activity='all', subjects=train_subjects,
                                                                correctness=correctness)

                print("Validation samples")
                X_valid_cross, y_valid_cross = generate_dataset(df=data_train_cross, n_time_steps=n_time_steps, n_features=n_features,
                                                    step=step, features=features,
                                                    one_vs_all_activity='all', subjects=valid_subjects,
                                                    correctness=correctness)

                print("Test samples")
                X_test_cross, y_test_cross = generate_dataset(df=data_test_cross, n_time_steps=n_time_steps, n_features=n_features,
                                                  step=step, features=features,
                                                  one_vs_all_activity='all', subjects=[left_out_subject],
                                                  correctness=correctness)

                X_train = np.concatenate((X_train, X_train_cross), axis=0)
                X_valid = np.concatenate((X_valid, X_valid_cross), axis=0)
                X_test = np.concatenate((X_test, X_test_cross), axis=0)
                y_train = np.concatenate((y_train, y_train_cross), axis=0)
                y_valid = np.concatenate((y_valid, y_valid_cross), axis=0)
                y_test = np.concatenate((y_test, y_test_cross), axis=0)

                print("Lengths of datasets = {}, {}, {}".format(len(X_train), len(X_valid), len(X_test)))
                print("Lengths of labels = {}, {}, {}".format(len(y_train), len(y_valid), len(y_test)))


        if normalisation != 'cross':
            print("-" * 80)
            print("Generating datasets")
            print("-" * 80)

            # Generate datasets
            X_train, y_train = generate_dataset(df=data_train, n_time_steps=n_time_steps, n_features=n_features,
                                                step=step, features=features,
                                                one_vs_all_activity='all', subjects=train_subjects, correctness=correctness)

            X_valid, y_valid = generate_dataset(df=data_train, n_time_steps=n_time_steps, n_features=n_features,
                                                step=step, features=features,
                                                one_vs_all_activity='all', subjects=valid_subjects, correctness=correctness)

            X_test, y_test = generate_dataset(df=data_test, n_time_steps=n_time_steps, n_features=n_features,
                                              step=step, features=features,
                                              one_vs_all_activity='all', subjects=[left_out_subject],
                                              correctness=correctness)

        # create new model
        print("n_classes = {}".format(n_classes))
        model = initialise_model(num_filters=num_filters, kernel_size=kernel_size, activation=activation,
                                 n_time_steps=n_time_steps, n_features=n_features, n_classes=n_classes,
                                 architecture=architecture, l1=l1, l2=l2, dropout_rate=dropout_rate)

        # compile model
        if optimiser == 'sgd':
            model.compile(optimizer=sgd,
                          loss='categorical_crossentropy',
                          metrics=['accuracy', 'mse'])
        elif optimiser == 'adam':
            model.compile(optimizer=adam,
                          loss='categorical_crossentropy',
                          metrics=['accuracy', 'mse'])

        history = model.fit(X_train, y_train,
                            batch_size=batch_size, epochs=epochs,
                            validation_data=(X_valid, y_valid))

        # stats
        y_pred_ohe = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred_ohe, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        print("*" * 80)
        print("Classification report")
        print("*" * 80)
        print(metrics.classification_report(y_true_labels, y_pred_labels))

        activity_labels = get_activity_label_dict()
        cm_labels = [activity_labels[i] for i in range(10)]

        confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
        fig = plt.figure(figsize=(10, 6))
        ax = sns.heatmap(confusion_matrix, xticklabels=cm_labels, yticklabels=cm_labels, annot=True, fmt='d')
        plt.xlabel("Predicted Labels", fontsize=14)
        plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
        plt.ylabel("True labels", fontsize=14)
        plt.title("CM - LOS = {}".format(left_out_subject))
        fig.tight_layout()
        plt.savefig(save_path + "{}_CM_LOS.pdf".format(left_out_subject))

        fig = plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title("Learning curve - LOS = {}".format(left_out_subject))
        plt.legend(fontsize=12)
        fig.tight_layout()
        plt.savefig(save_path + "{}_LC_LOS.pdf".format(left_out_subject))

        # record stats
        losoxv_stats.loc[left_out_subject, 'train_acc'] = history.history['accuracy'][-1]
        losoxv_stats.loc[left_out_subject, 'train_loss'] = history.history['loss'][-1]
        losoxv_stats.loc[left_out_subject, 'valid_acc'] = history.history['val_accuracy'][-1]
        losoxv_stats.loc[left_out_subject, 'valid_loss'] = history.history['val_loss'][-1]

        final_loss, final_acc, final_mse = model.evaluate(X_test, y_test, batch_size)
        print("Final loss = {}".format(final_loss))
        print("Final accuracy = {}".format(final_acc))

        losoxv_stats.loc[left_out_subject, 'test_acc'] = final_acc
        losoxv_stats.loc[left_out_subject, 'test_loss'] = final_loss

        # update cm and history
        cm[left_out_subject] = confusion_matrix
        histories[left_out_subject] = history

        # save everything
    losoxv_stats.to_csv("../Plots/{}/losoxv_stats.csv".format(experiment_name))
    pickle.dump(cm, open("../Plots/{}/cms.p".format(experiment_name), 'wb'))
    pickle.dump(histories, open("../Plots/{}/histories.p".format(experiment_name), 'wb'))

    # mean values of CM
    mean_cm = np.zeros(cm['adela'].shape)
    for subj in subjects:
        mean_cm += cm[subj]

    mean_cm /= len(subjects)

    fig = plt.figure(figsize=(10, 6))
    ax = sns.heatmap(mean_cm, xticklabels=cm_labels, yticklabels=cm_labels, annot=True)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True labels", fontsize=14)
    fig.tight_layout()
    plt.savefig("../Plots/{}/mean_cm_losoxv.pdf".format(experiment_name))

    # training and validation lines
    training_lines = []
    validation_lines = []

    for subj in subjects:
        training_lines.append(histories[subj].history['accuracy'])
        validation_lines.append(histories[subj].history['val_accuracy'])

    training_lines = np.asarray(training_lines)
    validation_lines = np.asarray(validation_lines)

    # mean and std of the learning curves
    training_mean = np.mean(training_lines, axis=0)
    validation_mean = np.mean(validation_lines, axis=0)

    training_std = np.std(training_lines, axis=0)
    validation_std = np.std(validation_lines, axis=0)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(training_mean, label='Training')
    plt.plot(validation_mean, '--', label='Validation')

    # Draw bands
    plt.fill_between(range(epochs), training_mean - training_std, training_mean + training_std, color="paleturquoise",
                     alpha=0.4)
    plt.fill_between(range(epochs), validation_mean - validation_std, validation_mean + validation_std,
                     color="moccasin", alpha=0.4)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim((0, 1))
    plt.legend(fontsize=14)

    fig.tight_layout()
    plt.savefig("../Plots/{}/mean_LC_losoxv.pdf".format(experiment_name))

    return losoxv_stats, cm, histories