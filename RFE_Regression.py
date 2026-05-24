import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score


class RFE_reg():

    # =========================
    # Split + Scaling
    # =========================
    @staticmethod
    def split_scalar(indep_X, dep_Y):

        X_train, X_test, y_train, y_test = train_test_split(
            indep_X,
            dep_Y,
            test_size=0.25,
            random_state=0
        )

        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        return X_train, X_test, y_train, y_test

    # =========================
    # R2 Prediction
    # =========================
    @staticmethod
    def r2_prediction(regressor, X_test, y_test):

        y_pred = regressor.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        return r2

    # =========================
    # Linear Regression
    # =========================
    @staticmethod
    def Linear(X_train, y_train, X_test, y_test):

        regressor = LinearRegression()

        regressor.fit(X_train, y_train)

        r2 = RFE_reg.r2_prediction(
            regressor,
            X_test,
            y_test
        )

        return r2

    # =========================
    # SVM Linear
    # =========================
    @staticmethod
    def svm_linear(X_train, y_train, X_test, y_test):

        regressor = SVR(kernel='linear')

        regressor.fit(X_train, y_train)

        r2 = RFE_reg.r2_prediction(
            regressor,
            X_test,
            y_test
        )

        return r2

    # =========================
    # SVM Non Linear
    # =========================
    @staticmethod
    def svm_NL(X_train, y_train, X_test, y_test):

        regressor = SVR(kernel='rbf')

        regressor.fit(X_train, y_train)

        r2 = RFE_reg.r2_prediction(
            regressor,
            X_test,
            y_test
        )

        return r2

    # =========================
    # Decision Tree
    # =========================
    @staticmethod
    def Decision(X_train, y_train, X_test, y_test):

        regressor = DecisionTreeRegressor(
            random_state=0
        )

        regressor.fit(X_train, y_train)

        r2 = RFE_reg.r2_prediction(
            regressor,
            X_test,
            y_test
        )

        return r2

    # =========================
    # Random Forest
    # =========================
    @staticmethod
    def random(X_train, y_train, X_test, y_test):

        regressor = RandomForestRegressor(
            n_estimators=10,
            random_state=0
        )

        regressor.fit(X_train, y_train)

        r2 = RFE_reg.r2_prediction(
            regressor,
            X_test,
            y_test
        )

        return r2

    # =========================
    # RFE Feature Selection
    # =========================
    @staticmethod
    def rfeFeature(indep_X, dep_Y, n):

        rfelist = []

        lin = LinearRegression()

        svrl = SVR(kernel='linear')

        dec = DecisionTreeRegressor(random_state=0)

        rf = RandomForestRegressor(
            n_estimators=10,
            random_state=0
        )

        rfemodellist = [lin, svrl, dec, rf]

        for model in rfemodellist:

            print(model)

            rfe = RFE(
                estimator=model,
                n_features_to_select=n
            )

            fit = rfe.fit(indep_X, dep_Y)

            transformed = fit.transform(indep_X)

            rfelist.append(transformed)

        return rfelist

    # =========================
    # Result DataFrame
    # =========================
    @staticmethod
    def rfe_regression(
            acclin,
            accsvml,
            accsvmnl,
            accdes,
            accrf):

        rfedataframe = pd.DataFrame(
            index=['Linear', 'SVMl', 'DecisionTree', 'RandomForest'],
            columns=['Linear', 'SVMl',
                     'SVMnl', 'Decision', 'Random']
        )

        for number, idx in enumerate(rfedataframe.index):

            rfedataframe.loc[idx, 'Linear'] = acclin[number]

            rfedataframe.loc[idx, 'SVMl'] = accsvml[number]

            rfedataframe.loc[idx, 'SVMnl'] = accsvmnl[number]

            rfedataframe.loc[idx, 'Decision'] = accdes[number]

            rfedataframe.loc[idx, 'Random'] = accrf[number]

        return rfedataframe

    # =========================
    # Main Function
    # =========================
    @staticmethod
    def Result():

        # Load Dataset
        dataset1 = pd.read_csv(
            "prep.csv",
            index_col=None
        )

        df2 = pd.get_dummies(
            dataset1,
            drop_first=True
        )

        indep_X = df2.drop(
            'classification_yes',
            axis=1
        )

        dep_Y = df2['classification_yes']

        # RFE Features
        rfelist = RFE_reg.rfeFeature(
            indep_X,
            dep_Y,
            4
        )

        # Accuracy Lists
        acclin = []
        accsvml = []
        accsvmnl = []
        accdes = []
        accrf = []

        # Training Loop
        for i in rfelist:

            X_train, X_test, y_train, y_test = \
                RFE_reg.split_scalar(i, dep_Y)

            r2_lin = RFE_reg.Linear(
                X_train,
                y_train,
                X_test,
                y_test
            )

            acclin.append(r2_lin)

            r2_sl = RFE_reg.svm_linear(
                X_train,
                y_train,
                X_test,
                y_test
            )

            accsvml.append(r2_sl)

            r2_nl = RFE_reg.svm_NL(
                X_train,
                y_train,
                X_test,
                y_test
            )

            accsvmnl.append(r2_nl)

            r2_d = RFE_reg.Decision(
                X_train,
                y_train,
                X_test,
                y_test
            )

            accdes.append(r2_d)

            r2_r = RFE_reg.random(
                X_train,
                y_train,
                X_test,
                y_test
            )

            accrf.append(r2_r)

        # Final Result
        result = RFE_reg.rfe_regression(
            acclin,
            accsvml,
            accsvmnl,
            accdes,
            accrf
        )

        print(result)

        return result