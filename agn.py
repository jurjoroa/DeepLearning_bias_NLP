import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score
from IPython import display
import numpy as np
np.random.seed(7)
import pandas as pd
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from functions.py_functions import plot_metric, plot_distributions, nn_classifier, p_rule, FairClassifier



df_compas = pd.read_csv('data_set.csv')


# One Hot encoding for categorical variable
charge_degree = pd.get_dummies(df_compas['charge_degree'])


charge_degree.columns = ['charge_degree_' + str(x) for x in charge_degree.columns]
df_compas = pd.concat([df_compas, charge_degree], axis=1)
# drop old label
df_compas.drop(['charge_degree'], axis=1, inplace=True)

# Inputs needed for training: 
# PREDICTORS 
X = df_compas.copy() # start from all variables and drop what is not needed during training

# Sensible attributes (we want to exclude them from training to avoid "intentional" bias)
Z_race = X.pop('race') # race not considered in training
Z_sex = X.pop('sex') # sex not considered in training
Z_data = {'race': Z_race, 'sex': Z_sex}
Z = pd.concat(Z_data, axis = 1)

# Target: COMPAS risk prediction, 1 = At risk of recidivism, 0 = No risk 
y = X.pop('target')



# Actual observed criminal activity reported within 2 years from compas score,
# 1 = the person committed a crime (he/she's a recidivist)
# 0 = he/she is not a recidivist
y_factual = X.pop('two_year_recid')
X.head()



np.random.seed(7)
X_train, X_test, y_train, y_test, y_factual_train, y_factual_test, Z_train, Z_test = train_test_split(X, y, y_factual, Z, test_size = 0.4, 
                                                                    stratify = y, random_state = 7)
# Normalize the data
scaler = MinMaxScaler().fit(X_train)
scale_df = lambda df_compas, scaler: pd.DataFrame(scaler.transform(df_compas), columns = df_compas.columns, index = df_compas.index)
X_train = X_train.pipe(scale_df, scaler) 
X_test = X_test.pipe(scale_df, scaler) 

def nn_classifier(n_features):
    inputs = Input(shape=(n_features,))
    dense1 = Dense(40, activation='relu')(inputs)
    dropout1 = Dropout(0.4)(dense1)  # Adjusted dropout rate
    dense2 = Dense(40, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    dense3 = Dense(32, activation='relu')(dropout2)
    dropout3 = Dropout(0.3)(dense3)  # Adjusted dropout rate
    outputs = Dense(1, activation='sigmoid')(dropout3)
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = Adam(learning_rate=0.001)  # Use learning_rate instead of lr
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

X.shape

### FIT THE MODEL
np.random.seed(7)
clf_1 = nn_classifier(n_features = X_train.shape[1])
history = clf_1.fit(X_train, y_train.values, epochs = 50, verbose = 2,validation_data = (X_test, y_test))

plot_metric(history, 'loss')

plot_metric(history, 'accuracy')

#check what history contains

X_train.shape


# PREDICTIONS
y_pred_1 = pd.Series(clf_1.predict(X_test).ravel(), index=y_test.index)
print(f"Accuracy: {100*accuracy_score(y_test, (y_pred_1>0.5)):.1f}%")

plot_distributions(y_pred_1, Z_test)


#show plot
plt.show()


print("The classifier satisfies the following %p-rules:")
print(f"\tgiven attribute race; {p_rule(y_pred_1, Z_test['race']):.0f}%-rule")
print(f"\tgiven attribute sex;  {p_rule(y_pred_1, Z_test['sex']):.0f}%-rule")

from sklearn.utils.class_weight import compute_class_weight




class FairClassifier(object):

    def __init__(self, n_features, n_sensitive, lambdas):
        self.lambdas = lambdas

        clf_inputs = Input(shape=(n_features,))
        adv_inputs = Input(shape=(1,))

        clf_net = self._create_clf_net(clf_inputs)
        adv_net = self._create_adv_net(adv_inputs, n_sensitive)
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(clf_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(clf_inputs, clf_net, adv_net, n_sensitive)
        self._val_metrics = None
        self._fairness_metrics = None
        self.predict = self._clf.predict

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag
        return make_trainable

# Generator (= Feed Forward built before)
    def _create_clf_net(self, inputs):
        dense1 = Dense(40, activation='relu')(inputs)
        dropout1 = Dropout(0)(dense1)
        dense2 = Dense(40, activation='relu')(dropout1)
        dropout2 = Dropout(0.1)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        dropout3 = Dropout(0)(dense3)
        outputs = Dense(1, activation='sigmoid', name='y')(dropout3)
        return Model(inputs=[inputs], outputs=[outputs])

# Discriminator / Adversarial 
    def _create_adv_net(self, inputs, n_sensitive):
        dense1 = Dense(32, activation='relu')(inputs)
        dense2 = Dense(32, activation='relu')(dense1)
        dense3 = Dense(32, activation='relu')(dense2)
        outputs = [Dense(1, activation='sigmoid')(dense3) for _ in range(n_sensitive)]
        return Model(inputs=[inputs], outputs=outputs)

    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        return clf

    def _compile_clf_w_adv(self, inputs, clf_net, adv_net):
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs)]+adv_net(clf_net(inputs)))
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.]+[-lambda_param for lambda_param in self.lambdas]
        clf_w_adv.compile(loss=['binary_crossentropy']*(len(loss_weights)),
                        loss_weights=loss_weights,
                        optimizer='adam')
        return clf_w_adv

    def _compile_adv(self, inputs, clf_net, adv_net, n_sensitive):
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        adv.compile(loss=['binary_crossentropy']*n_sensitive, optimizer='adam')
        return adv

    def _compute_class_weights(self, data_set):
        class_values = [0, 1]
        class_weights = []
        if len(data_set.shape) == 1:
            balanced_weights = compute_class_weight(class_weight='balanced', 
                                                    classes=class_values, 
                                                    y=data_set)
            class_weights.append(dict(zip(class_values, balanced_weights)))
        else:
            n_attr = data_set.shape[1]
            for attr_idx in range(n_attr):
                balanced_weights = compute_class_weight(class_weight='balanced', 
                                                        classes=class_values,
                                                        y=np.array(data_set)[:, attr_idx])
                class_weights.append(dict(zip(class_values, balanced_weights)))
        return class_weights

    def _compute_target_class_weights(self, y):
        class_values = [0,1]
        balanced_weights = compute_class_weight('balanced', class_values, y)
        class_weights = {'y': dict(zip(class_values, balanced_weights))}
        return class_weights

    def pretrain(self, x, y, z, epochs=10, verbose=0):
        self._trainable_clf_net(True)
        self._clf.fit(x, y, epochs=epochs, verbose=verbose)
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        class_weight_adv = self._compute_class_weights(z)
        self._adv.fit(x, np.hsplit(z, z.shape[1]), class_weight=class_weight_adv,
                        epochs=epochs, verbose=verbose)

    def fit(self, x, y, z, validation_data=None, T_iter=250, batch_size=128,
            weight_sensitive_classes=True, save_figs=True):
        n_sensitive = z.shape[1]
        if validation_data is not None:
            x_val, y_val, z_val = validation_data
        class_weight_adv = self._compute_class_weights(z)
        class_weight_clf_w_adv = [{0:1.,1:1.}]+class_weight_adv
        self._val_metrics = pd.DataFrame()
        self._fairness_metrics = pd.DataFrame()
        for idx in range(T_iter):
            if validation_data is not None:
                y_pred = pd.Series(self._clf.predict(x_val).ravel(), index=y_val.index)
                self._val_metrics.loc[idx, 'ROC AUC'] = roc_auc_score(y_val, y_pred)
                self._val_metrics.loc[idx, 'Accuracy'] = (accuracy_score(y_val, (y_pred>0.5))*100)
                for sensitive_attr in z_val.columns:
                    self._fairness_metrics.loc[idx, sensitive_attr] = p_rule(y_pred,
                                                                            z_val[sensitive_attr])
                display.clear_output(wait=True)
                plot_distributions(y_pred, z_val, idx+1, self._val_metrics.loc[idx],
                                    self._fairness_metrics.loc[idx])
                if save_figs:
                    plt.savefig(f'output/{idx+1:08d}.png', bbox_inches='tight')
                plt.show(plt.gcf())

            # train adverserial
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            self._adv.fit(x, np.hsplit(z, z.shape[1]), batch_size=batch_size,
                            class_weight=class_weight_adv,
                            epochs=1, verbose=0)

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            self._clf_w_adv.train_on_batch(x.values[indices],
                                            [y.values[indices]]+np.hsplit(z.values[indices],
                                                                        n_sensitive),
                                                                        class_weight=class_weight_clf_w_adv)



clf_2 = FairClassifier(n_features = X_train.shape[1], n_sensitive = Z_train.shape[1], lambdas = [130., 30.])

# pre-train both adverserial and classifier networks
clf_2.pretrain(X_train, y_train, Z_train, verbose=2, epochs=10)

history_2 = clf_2.fit(X_train, y_train, Z_train, validation_data = (X_test, y_test, Z_test), T_iter = 82, save_figs = False)