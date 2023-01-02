# Project 7: Implement a scoring model
# Import libraries
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

# Serialisation library
import pickle

# Front end library
import streamlit as st

# SHAP library
import shap

# Visualisation library
import plotly_express as px


from sklearn.neighbors import NearestNeighbors
plt.style.use('seaborn')


def load_data(file):
    """This function is used to load the dataset."""
    folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(folder, file)
    data = pd.read_csv(data_path, encoding_errors='ignore')
    return data


def preprocessing(data, num_imputer, bin_imputer, transformer, scaler):
    """This function is used to perform data preprocessing."""
    X_df = data.drop(['SK_ID_CURR'], axis=1)

    # Feature selection
    # Categorical features
    cat_features = list(data.select_dtypes('object').nunique().index)

    # Encoding the categorical features
    df = pd.get_dummies(X_df, columns=cat_features)

    # Numerical and binary features
    features_df = df.nunique()
    num_features = list(features_df[features_df != 2].index)
    binary_features = list(features_df[features_df == 2].index)
    df['NAME_FAMILY_STATUS_Unknown'] = 0
    binary_features.append('NAME_FAMILY_STATUS_Unknown')

    # Imputations
    X_num = pd.DataFrame(num_imputer.transform(df[num_features]),
                         columns=num_features)
    X_bin = pd.DataFrame(bin_imputer.transform(df[binary_features]),
                         columns=binary_features)

    # Normalisation
    X_norm = pd.DataFrame(transformer.transform(X_num), columns=num_features)

    # Standardisation
    norm_df = pd.DataFrame(scaler.transform(X_norm), columns=num_features)

    for feature in binary_features:
        norm_df[feature] = X_bin[feature]
    norm_df['SK_ID_CURR'] = data['SK_ID_CURR']
    return norm_df


def load_model(file, key):
    """This function is used to load a serialised file."""
    path = open(file, 'rb')
    model_pickle = pickle.load(path)
    model = model_pickle[key]
    return model


def predictor(X, model, threshold):
    """This function is used for making prediction.
    and returns the score, the situation,
    and the status of the customer's application"""
    data_json = X.to_dict(orient="records")[0]
    data = []
    for key in data_json.keys():
        data.append(data_json[key])

    # Making prediction
    y_proba = model.predict_proba([data])[0][0]

    # Finding the situation of the customer (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = threshold * 0.01
    customer_class = np.where(y_class > best_threshold, 1, 0)

    # Calculation of the customer's score
    score = int(y_class * 100)

    # Result of the credit application
    if customer_class == 1:
        situation = 'at risk'
        status = 'refused'
    else:
        situation = 'without risk'
        status = 'granted'
    return score, situation, status


def customer_description(data):
    """This function creates a dataframe with
     the description of the customer."""
    df = pd.DataFrame(
        columns=['Gender', 'Age (years)', 'Family status',
                 'Number of children', 'Days employed',
                 'Income ($)', 'Credit amount ($)', 'Loan annuity ($)'])
    data['AGE'] = data['DAYS_BIRTH'] / 365
    df['Customer ID'] = list(data.SK_ID_CURR.astype(str))
    df['Gender'] = list(data.CODE_GENDER)
    df['Age (years)'] = list(data.AGE.abs().astype('int64'))
    df['Family status'] = list(data.NAME_FAMILY_STATUS)
    df['Number of children'] = list(data.CNT_CHILDREN.astype('int64'))
    df['Days employed'] = list(data.DAYS_EMPLOYED.abs().astype('int64'))
    df['Income ($)'] = list(data.AMT_INCOME_TOTAL.astype('int64'))
    df['Credit amount ($)'] = list(data.AMT_CREDIT.astype('int64'))
    df['Loan annuity ($)'] = list(data.AMT_ANNUITY.astype('int64'))
    df['Organization type'] = list(data.ORGANIZATION_TYPE)
    return df


def apply_knn(X, X_norm, data, features):
    """This function uses the near neighbors' algorithm
    to find the most similar group of a customer.
    """
    X_norm = X_norm[features]
    X = X[features]
    neigh = NearestNeighbors(
        n_neighbors=11,
        leaf_size=30,
        metric='minkowski',
        p=2)
    neigh.fit(X_norm)
    indice = neigh.kneighbors(X, return_distance=False)
    index_list = list(indice[0])
    knn_df = data.iloc[index_list, :]
    return knn_df


def main():
    st.set_page_config(layout='wide')
    st.title("CREDIT SCORING DASHBOARD")

    # Loading the dataset
    data = load_data('data/data.csv')

    # Loading the model
    model = load_model('model/model.pkl', 'model')

    # Loading the numerical imputer
    num_imputer = load_model('model/num_imputer.pkl', 'num_imputer')

    # Loading the binary imputer
    bin_imputer = load_model('model/bin_imputer.pkl', 'bin_imputer')

    # Loading the numerical transformer
    transformer = load_model('model/transformer.pkl', 'transformer')

    # Loading the numerical scaler
    scaler = load_model('model/scaler.pkl', 'scaler')

    # Preprocessing
    norm_df = preprocessing(data,
                            num_imputer,
                            bin_imputer,
                            transformer,
                            scaler)
    X_norm = norm_df.drop(['SK_ID_CURR'], axis=1)

    # Selection of the customer
    customers_list = list(data.SK_ID_CURR)
    customer_id = st.sidebar.selectbox(
        "Select or enter a customer ID:", customers_list)

    # Selection of the threshold
    # The default value of the threshold is 36
    best_threshold = 36

    # The threshold varies between 0 and 100
    selected_threshold = st.sidebar.slider(
        "Select the threshold value:", min_value=0, max_value=100)
    if selected_threshold <= 0 or selected_threshold >= 100:
        # The default value is applied
        # if the selected value is not between 0 and 100
        threshold = best_threshold
    else:
        threshold = selected_threshold

    # Retrieving the customer's data
    customer_df = data[data.SK_ID_CURR == customer_id]
    viz_df = customer_df.round(2)

    # Preprocessing the data of the customer for the prediction
    X = norm_df[norm_df.SK_ID_CURR == customer_id]
    X = X.drop(['SK_ID_CURR'], axis=1)

    # Results of the prediction
    score, situation, status = predictor(X, model, threshold)
    st.header("Status of the credit application")
    st.write("The credit score varies between 0 and 100. "
             "According to the model evaluation, the best value of "
             "the threshold is {}. This is the default value. That is why, "
             "customers with scores above 36 are at risk.".format(threshold))
    st.write("**The score of the customer N°{} is {}.** "
             "The customer's situation is {}. So,"
             " the credit application status is {}.".format(customer_id, score,
                                                            situation, status))

    # Feature Importance
    model.predict(np.array(X_norm))
    features_importance = model.feature_importances_
    sorted = np.argsort(features_importance)
    dataviz = pd.DataFrame(columns=['feature', 'importance'])
    dataviz['feature'] = np.array(X_norm.columns)[sorted]
    dataviz['importance'] = features_importance[sorted]
    dataviz = dataviz[dataviz['importance'] > 0]
    dataviz.reset_index(inplace=True, drop=True)
    dataviz = dataviz.sort_values(['importance'], ascending=False)

    # SHAP explanations
    shap.initjs()
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X)
    shap_df = pd.DataFrame(
        list(zip(X.columns, np.abs(shap_values[0]).mean(0))),
        columns=['feature', 'importance'])
    shap_df = shap_df.sort_values(by=['importance'], ascending=False)
    shap_df.reset_index(inplace=True, drop=True)
    shap_features = list(shap_df.iloc[0:20, ].feature)

    # Description of the customer
    st.header("Descriptive information of the customer")
    info_viz = customer_description(customer_df)
    st.dataframe(info_viz.set_index('Customer ID'))

    # Information of the customer
    info_display = st.sidebar.selectbox(
        "Select the topic to display:",
        ["Visualisations",
         "Similar customers",
         "Global interpretability of the model",
         "Local interpretability of the model",
         "Customer's data"])

    # Selecting the features for the descriptive information
    features = ['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']
    for feature in shap_features:
        if feature not in features:
            features.append(feature)

    # Applying the Nearest Neighbors function
    with st.spinner("Loading..."):
        knn_df = apply_knn(X, X_norm, data, features)
    group_df = knn_df[knn_df.SK_ID_CURR != customer_id]
    group_viz = customer_description(group_df)
    infos_viz = info_viz.append(group_viz)

    if info_display == "Visualisations":
        st.header("Visualisations of the descriptive information")
        st.write("Visualisations are used to compare"
                 " the descriptive information of the customer N°{}"
                 " with 10 similar clients.".format(customer_id))
        display_description = st.sidebar.selectbox(
            "Select the descriptive information:",
            ["Financial information", "Gender", "Age",
             "Professional status", "Number of children",
             "Family status", "Business segment"])
        if display_description == "Financial information":
            st.subheader("Financial information")
            fig1 = px.bar(infos_viz, x='Customer ID',
                          y=['Credit amount ($)',
                             'Income ($)',
                             'Loan annuity ($)'],
                          height=500)
            st.write(fig1)
            financial_features = st.multiselect(
                label="Select the financial information to be displayed:",
                options=['Credit amount ($)',
                         'Income ($)',
                         'Loan annuity ($)'])
            time.sleep(5)
            df_viz = infos_viz[['Credit amount ($)',
                                'Income ($)',
                                'Loan annuity ($)']]
            df_viz['Customer ID'] = list(knn_df.SK_ID_CURR)
            viz = df_viz.set_index('Customer ID')
            if not financial_features:
                st.info("Sorry, no information is given!")
            else:
                st.dataframe(viz[financial_features])
                ax = viz[financial_features].plot(kind='bar')
                fig2 = ax.get_figure()
                st.pyplot(fig2)
        elif display_description == "Gender":
            st.subheader("Gender")
            data_viz = infos_viz.groupby('Gender').count()
            data_viz.reset_index(inplace=True, drop=False)
            data_viz = data_viz.rename(columns={"Customer ID": "Count"})
            fig3 = px.bar(data_viz, x='Gender', y='Count', height=300)
            st.write(fig3)
        elif display_description == "Age":
            st.subheader("Age")
            radar_df = pd.DataFrame({
                'Age': list(infos_viz['Age (years)']),
                'Customer ID': list(infos_viz['Customer ID'])})
            fig4 = px.line_polar(radar_df,
                                 r='Age',
                                 theta='Customer ID',
                                 line_close=True)
            fig4.update_traces(fill='toself')
            st.write(fig4)
        elif display_description == "Professional status":
            st.subheader("Professional status")
            st.write("We see below the number of days elapsed"
                     " since the beginning of the last employment contract.")
            radar_df = pd.DataFrame({
                'Days employed': list(infos_viz['Days employed']),
                'Customer ID': list(infos_viz['Customer ID'])})
            fig5 = px.line_polar(radar_df,
                                 r='Days employed',
                                 theta='Customer ID',
                                 line_close=True)
            fig5.update_traces(fill='toself')
            st.write(fig5)
        elif display_description == "Number of children":
            st.subheader("Number of children")
            radar_df = pd.DataFrame({
                'Number of children': list(infos_viz['Number of children']),
                'Customer ID': list(infos_viz['Customer ID'])})
            fig6 = px.line_polar(radar_df,
                                 r='Number of children',
                                 theta='Customer ID',
                                 line_close=True)
            fig6.update_traces(fill='toself')
            st.write(fig6)
        elif display_description == "Family status":
            st.subheader("Marital status")
            viz_data = infos_viz.groupby('Family status').count()
            viz_data.reset_index(inplace=True, drop=False)
            viz_data = viz_data.rename(columns={"Customer ID": "Count"})
            fig7 = px.bar(viz_data, x='Family status', y='Count', height=400)
            st.write(fig7)
        elif display_description == "Business segment":
            st.subheader("Business segment")
            viz_data = infos_viz.groupby('Organization type').count()
            viz_data.reset_index(inplace=True, drop=False)
            viz_data = viz_data.rename(columns={"Customer ID": "Count"})
            fig8 = px.bar(
                viz_data, x='Organization type', y='Count', height=500)
            st.write(fig8)
    elif info_display == "Similar customers":
        st.header("Similar customers")
        st.write("Grouping allows us to compare the customer N°{}"
                 " with 10 similar customers.".format(customer_id))
        st.write("This grouping is based on the descriptive information"
                 " and the important data for the prediction of score"
                 " (see the local interpretability of the model).")
        st.dataframe(group_viz.set_index('Customer ID'))

        # Selection of the similar customer
        clients_list = list(group_df.SK_ID_CURR)
        client_id = st.sidebar.selectbox(
            "Select or enter a similar customer ID:", clients_list)

        # Preprocessing the data of the similar customers for the prediction
        X_df = norm_df[norm_df.SK_ID_CURR == client_id]
        X_df = X_df.drop(['SK_ID_CURR'], axis=1)

        # Results of the prediction
        score, situation, status = predictor(X_df, model, threshold)
        st.write("**The score of the customer N°{} is {}.** "
                 "The customer's situation is {}. So, the credit application"
                 " status is {}.".format(client_id, score, situation, status))
    elif info_display == "Global interpretability of the model":
        st.header("Global interpretability of the model")
        fig9 = plt.figure(figsize=(10, 20))
        sns.barplot(x='importance', y='feature', data=dataviz)
        st.write("The GDPR (Article 22) provides restrictive rules"
                 " to prevent human from being subjected to decisions"
                 " emanating only from machines.")
        st.write("The global interpretability provides a general understanding"
                 " of the important features for the model. ")
        st.write("The feature importance doesn't change following to"
                 " each customer's data.")
        st.write(fig9)
    elif info_display == "Local interpretability of the model":
        st.header("Local interpretability of the model")
        fig10 = plt.figure()
        shap.summary_plot(shap_values, X,
                          feature_names=list(X.columns),
                          max_display=50,
                          plot_type='bar',
                          plot_size=(5, 15))
        st.write("The GDPR (Article 22) provides restrictive rules"
                 " to prevent human from being subjected to decisions"
                 " emanating only from machines.")
        st.write("SHAP meets the requirements of the GDPR and allows us"
                 " to determine the effects of the features"
                 " in the result of predicting the customer's N°{} score. "
                 "The feature importance changes following to"
                 " each customer's data.".format(customer_id))
        st.pyplot(fig10)
    elif info_display == "Customer's data":
        st.header("Important data of the customer")
        st.write("Displaying the important data of the customer for"
                 " the prediction of score.")
        viz_df = viz_df.astype('str')
        viz_df['Data'] = 'Data'
        viz_df.set_index('Data', inplace=True)
        viz_df = viz_df.transpose()
        st.dataframe(viz_df)

        # Loading the description of the dataset
        st.subheader("Description of the data")
        st.write("Displaying the description of the customer's data.")
        data_description = load_data('data/data_columns_description.csv')
        st.dataframe(data_description.set_index('Row'))


if __name__ == '__main__':
    main()
