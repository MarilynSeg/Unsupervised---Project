# importing relevant libraries
import openpyxl
import pandas as pd
import numpy
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Inserting image
st.image("ass.image.jpeg", width=200)

# Import data at the global level
data = pd.read_excel("country.xlsx")

# selecting the attributes....remove the attribute call country
select_att = data[
    ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17']]

# Scaling the dataset for analysis.
scaled = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaled.fit_transform(select_att)

# this codes adds column heading to the dataset after scaling
column_header = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
                 'V17']
scaled_data = pd.DataFrame(scaled_dataset, columns=column_header)

# Group Members Names
def groupC_members():
    st.write("### Group Members")
    '''Marilyn Nicco-Annan_11410745'''
    '''Suleman Abdul-Razark_22256374'''
    '''Priscilla Oteng Asamoah_22252463'''
    '''Jude Kwaku Afenyo Gbeddie_22253326'''

# Creating functions and segment them in selectbox
def page1():
    st.header('Project description')
    '''This project is based off on the journal paper “Classification of European countries according to  
indicators related to electricity generation” by the authors Alvaro Gonzalez-Lorente, Montserrat Hernandez-Lopez, Francisco Javier Martin-Alvarez Imanol L. Nieto-Gonzalez.  
The aim of the project is to analyse the differences and similarities between European countries in terms of electricity generation,using hierachical clustering and K-means clustering methods.       
The countries chosen are the 45 that geographically and politically belong to the European  
continent, 27 of which belong to the EU and abide by the same environmental policy and 18 that do not follow the common environmental policy'''

def section_a_q1():
    st.write("### Section_A Que.1")
    '''Answer'''
    '''The study "Classification of European Countries According to Indicators Related to Electricity Generation" effectively covers a number of crucial clustering topics, such as issue formulation, data preparation, technique selection, validation, interpretation, and visual depiction. There is room for improvement in a few areas, though. The analysis omits potentially important variables and only uses data from 2020, which may raise questions about generalizability. Furthermore, the study did not use external validation measurements or investigate alternate clustering techniques. However, by defining seven clusters that are distinguished by certain patterns in energy generation, consumption, and socioeconomic situations, the study offers insightful information about the patterns of electricity generation throughout Europe. The study notes its shortcomings and offers suggestions for further research, such as expanding the analysis to several years and investigating alternative clustering strategies. Furthermore, the study did not use external validation measurements or investigate alternate clustering techniques. However, by defining seven clusters that are distinguished by certain patterns in energy generation, consumption, and socioeconomic situations, the study offers insightful information about the patterns of electricity generation throughout Europe.  
     The study notes its shortcomings and offers suggestions for further research, such as expanding the analysis to several years and investigating alternative clustering strategies. Though it might use more methods and metrics to improve its robustness and comprehensiveness, the study shows a thorough approach to clustering overall.'''

def section_a_q2():
    st.write("### Section_A Que.2")
    '''Answer'''
    '''Using 17 variables linked to electricity generation, the publication "Classification of European Countries According to Indicators Related to Electricity Generation" provides a thorough clustering analysis of 45 European nations. Aspects of clustering such as problem creation, data preparation, procedure selection, validation, interpretation, and visual representation are all skillfully covered in the paper. However, the research leaves out potentially important variables and only uses data from 2020, which may raise questions regarding generalizability. Furthermore, the study did not use external validation measurements or investigate alternate clustering techniques. Notwithstanding these drawbacks, the study offers insightful information about the trends in electricity generation in Europe by defining seven clusters that are distinguished by a particular trend in energy production, consumption, and socioeconomic circumstances. The study notes its shortcomings and offers suggestions for further research, such as expanding the analysis to several years and investigating alternative clustering strategies. Though it might use more methods and metrics to improve its robustness and comprehensiveness, the study shows a thorough approach to clustering overall.'''

def section_a_q3():
    st.write("### Section_A Que.3")
    '''Answer'''
    '''The paper offers a thorough examination of trends in energy-related issues in various European nations. The study has many shortcomings, despite its methodological rigor and clear cluster interpretations. By investigating alternate clustering techniques and validating the findings, the analysis's reliance on hierarchical clustering and lack of external validation could be addressed. Furthermore, the study may oversimplify the dataset due to its dependence on single-year data and elimination of associated factors. The authors may have included more thorough policy recommendations catered to the particular opportunities and difficulties of each cluster to increase the study's impact. Interpretability would also be enhanced by adding statistical analysis and richer visualizations. By tackling these topics, the study may offer a more thorough and useful categorization of European nations according to their production of power, which would ultimately guide the development of a more efficient energy policy.'''

def section_a_q4():
    st.write("### Section_A Que.4")
    '''Answer'''
    '''The use of clustering in this study is appropriate for exploratory analysis and for classifying European nations according to metrics related to power generation. Its drawbacks, however, are its one-year data, excessive dependence on similarity criteria, and lack of predictive capacity. Alternative techniques including supervised learning, regression analysis, PCA, network analysis, and time-series analysis could be used to overcome these constraints and improve the results. Deeper understanding of the variables influencing patterns of power generation and their potential evolution would be possible through a multi-method approach that combines clustering with other techniques. The study might have produced more useful policy recommendations by combining several approaches to give a more thorough grasp of the underlying dynamics and linkages.'''

def section_a_q5():
    st.write("### Section_A Que.5")
    '''Answer'''
    '''Prior to clustering, exploratory data analysis (EDA) is essential for guaranteeing data quality, comprehending trends, and spotting possible problems. Data quality evaluation, distribution analysis, correlation analysis, feature scaling, variable importance analysis, multivariate analysis, dimensionality reduction, pattern visualization, and data subsetting are some of the stages that are included in EDA. Finding missing values, outliers, and inconsistent data, comprehending variable linkages, and visualizing patterns are all made possible by EDA, which eventually produces clustering findings that are more trustworthy and comprehensible. EDA is a crucial phase in the clustering process since it improves model performance, guides algorithm selection, and yields useful insights. A comprehensive EDA would have improved the study's foundation for clustering and given more insight into the patterns of power generation in European nations, which would have ultimately resulted in more sensible policy suggestions.'''

def page2(excel_file=None):
    if st.checkbox("Dataset"):  # displaying data when a checkbox is selected
        st.write(data)

    if st.checkbox('Selected attributes'):  # displaying selected attribute data when a checkbox is selected
        st.write(select_att)

    if st.checkbox("Scaled data"):  # displaying scaled data when a checkbox is selected
        st.write(scaled_data)

    # importing data from the web
    excel_file = st.file_uploader("Click to upload your excel file", type=['excel'])

    # Check if file is uploaded
    if excel_file is not None:
        # Read excel file
        df = pd.read_excel(excel_file)

        # Display dataset
        st.subheader("Uploaded Dataset:")
        st.write(df)

        # Display dataset statistics
        st.subheader("Dataset Statistics:")
        st.write(df.describe())

        # Display dataset information
        st.subheader("Dataset Information:")
        st.write(df.info())


def page3():
    st.header('Exploratory Data Analysis')
# Statistics for the selected attributes

    st.write("click on the check box to display")
    if st.checkbox('Descriptive of selected attributes'):
        st.write(select_att.describe())

    if st.checkbox('Heatmap of selected attributes'):
        cor = scaled_data.corr()

        #Heatmap showing the various correllation
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cor, annot=True, cmap='coolwarm', square=True)
        plt.title('Heatmap of selected Attributes')
        st.pyplot(fig3)

def page4():
    st.header('Variable Description')

    #Explaining the meanings of the 17 indicators
    st.subheader('Description of Variables used in cluster')
    '''V1 Mineral depletion WB Ratio of the value of the stock of mineral resources to the remaining reserve lifetime (capped at 25 years). It covers tin, gold, lead, zinc, iron, copper, nickel, silver, bauxite, and phosphate. Percentage of Gross National Income (GNI).   
    V2 Energy depletion WB Ratio of the value of the stock of energy resources to the remaining reserve lifetime (capped at 25 years). It covers coal, crude oil, and natural gas. Percentage of GNI.   
    V3 Net ODA received WB Net official development assistance (ODA) consists of disbursements of loans made on concessional terms (net of repayments of principal) and grants by official agencies of the members of the Development Assistance Committee (DAC), by multilateral institutions, and by non-DAC countries to promote economic development and welfare in countries and territories in the DAC list of ODA recipients. It includes loans with a grant element of at least 25% (calculated at a rate of discount of 10%). Percentage of GNI.   
    V4 Fuel exports WB Fuel exports (% of merchandise exports). Fuels comprise the commodities in SITC (Standard International Trade Classification) section 3 (mineral fuels, lubricants and related materials).   
    V5 Fuel imports WB Fuel imports (% of merchandise imports). Fuels comprise the commodities in SITC (Standard International Trade Classification) section 3 (mineral fuels, lubricants and related materials).    
    V6 GDP WB GDP per capita, PPP (constant 2017 international $).   
    V7 Population WB Total population.   
    V8 Per capita electricity consumption OWID It is measured in kilowatt/hour (kWh) per capita.   
    V9 Human Development Index UN Summary measure of average achievement in key dimensions of human development: Life expected at birth, years of schooling and GNI per capita.  
    V10 Coal electricity generation IEA It is measured in Gigawatt/hour (GWh).   
    V11 Oil electricity generation IEA It is measured in GWh.   
    V12 Natural gas electricity generation IEA It is measured in GWh.   
    V13 Nuclear electricity generation IEA It is measured in GWh.   
    V14 Hydroelectric electricity generation IEA It is measured in GWh.'''

def page5():
    st.header('Hierachical Clustering')

    # Agglomerative Clustering
    st.subheader("Agglomerative Clustering")

    # Define a list of options for the number of clusters
    cluster_options = [2, 3, 4, 5, 6, 8]

    # Create a selectbox widget to let the user choose the number of clusters
    n_clusters = st.selectbox("Select your preferred number of clusters", cluster_options)

    # Compute the pairwise distances between all samples in the scaled data
    distances = pdist(scaled_data, metric='euclidean')

    linkage_matrix = linkage(distances, method='ward')

    # Create an AgglomerativeClustering instance with the chosen number of clusters and 'ward' linkage
    Agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

    # Fit the AgglomerativeClustering instance to the scaled data
    Agg.fit(scaled_data)
    data['Agg Cluster'] = Agg.labels_  # Predict the cluster labels for each sample in the data
    st.write(data)

    '''Ward's clustering produces well-separated clusters and minimizes the within-cluster variance'''
    # Write a title for the dendrogram plot
    st.write("Dendrogram for Agglomerative Clustering,Ward")
    # Create a new figure and axis object for the dendrogram plot
    fig2, ax2 = plt.subplots()
    # Plot the dendrogram using the linkage matrix
    dendrogram(linkage_matrix, ax=ax2)
    # Set the title, x-axis label, and y-axis label for the dendrogram plot
    ax2.set_title("Dendrogram for Agglomerative Clustering,Ward")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Distance')
    st.pyplot(fig2)

    # Evaluation of the algorithm
    sil1 = silhouette_score(scaled_data, Agg.labels_)  # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is", sil1 * 100)

    # Compute the pairwise distances between all samples in the scaled data
    distances = pdist(scaled_data, metric='euclidean')

    linkage_matrix1 = linkage(distances, method='single')

    # Create an AgglomerativeClustering instance with the chosen number of clusters and 'single' linkage
    Agg_s = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')

    # Fit the AgglomerativeClustering instance to the scaled data
    Agg_s.fit(scaled_data)
    data['Agg Cluster'] = Agg_s.labels_  # Predict the cluster labels for each sample in the data
    st.write(data)

    '''Single linkage measures the distance between the closest points of two clusters,where clusters form long strings and produces chaining effects.'''
    # Write a title for the dendrogram plot
    st.write("Dendrogram for Agglomerative Clustering,Single")
    # Create a new figure and axis object for the dendrogram plot
    fig2, ax2 = plt.subplots()
    # Plot the dendrogram using the linkage matrix
    dendrogram(linkage_matrix1, ax=ax2)
    # Set the title, x-axis label, and y-axis label for the dendrogram plot
    ax2.set_title("Dendrogram for Agglomerative Clustering,Single")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Distance')
    st.pyplot(fig2)

    # Evaluation of the algorithm
    sil1 = silhouette_score(scaled_data, Agg.labels_)  # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is", sil1 * 100)

    # Compute the pairwise distances between all samples in the scaled data
    distances = pdist(scaled_data, metric='euclidean')

    linkage_matrix2 = linkage(distances, method='complete')

    # Create an AgglomerativeClustering instance with the chosen number of clusters and 'complete' linkage
    Agg_clust = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')

    # Fit the AgglomerativeClustering instance to the scaled data
    Agg_clust.fit(scaled_data)
    data['Agg Cluster'] = Agg_clust.labels_  # Predict the cluster labels for each sample in the data
    st.write(data)

    '''Complete linkage measures the distance between the farthest points in two clusters.It produces tight and compact clusters.'''
    # Write a title for the dendrogram plot
    st.write("Dendrogram for Agglomerative Clustering,Complete")
    # Create a new figure and axis object for the dendrogram plot
    fig2, ax2 = plt.subplots()
    # Plot the dendrogram using the linkage matrix
    dendrogram(linkage_matrix2, ax=ax2)
    # Set the title, x-axis label, and y-axis label for the dendrogram plot
    ax2.set_title("Dendrogram for Agglomerative Clustering,Complete")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Distance')
    st.pyplot(fig2)

    # Evaluation of the algorithm
    sil1 = silhouette_score(scaled_data, Agg_clust.labels_)  # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is", sil1 * 100)

    # Compute the pairwise distances between all samples in the scaled data
    distances = pdist(scaled_data, metric='euclidean')

    linkage_matrix3 = linkage(distances, method='average')

    # Create an AgglomerativeClustering instance with the chosen number of clusters and 'complete' linkage
    Agg1 = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')

    # Fit the AgglomerativeClustering instance to the scaled data
    Agg1.fit(scaled_data)
    data['Agg Cluster'] = Agg1.labels_  # Predict the cluster labels for each sample in the data
    st.write(data)

    '''Average linkage merges clusters based on the average distance between all points in the clusters.'''
    # Write a title for the dendrogram plot
    st.write("Dendrogram for Agglomerative Clustering,Average")
    # Create a new figure and axis object for the dendrogram plot
    fig2, ax2 = plt.subplots()
    # Plot the dendrogram using the linkage matrix
    dendrogram(linkage_matrix3, ax=ax2)
    # Set the title, x-axis label, and y-axis label for the dendrogram plot
    ax2.set_title("Dendrogram for Agglomerative Clustering,Average")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Distance')
    st.pyplot(fig2)

    # Evaluation of the algorithm
    sil1 = silhouette_score(scaled_data, Agg1.labels_)  # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is", sil1 * 100)
    st.write(
        "The evaluation score quantifies the quality of a clustering solution based on the distances between points within the same cluster and points in different clusters.The higher the score,the better the cluster.")

    #Compute the pairwise distances between all samples in the scaled data
    distances = pdist(scaled_data, metric='euclidean')

    linkage_matrix4 = linkage(distances, method='centroid')

    #create an AgglomerativeClustering instance with the chosen number of clusters and 'centroid' linkage
    Agg2 = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

    #Fit the AgglomerativeClustering instance to the scaled data
    Agg2.fit(scaled_data)
    data['Agg Cluster'] = Agg2.labels_
    st.write(data)

    #Write a title for the dendrogram plot
    st.write("Dendrogram for Agglomerative Clustering, Centroid")

    #Create a new figure and axis object for the dendrogram plot
    fig2, ax2 = plt.subplots()

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix4, ax=ax2)
    ax2.set_title("Dendrogram using Ward's Centroid Linkage")
    ax2.set_xlabel("Countries")
    ax2.set_ylabel("Distances")
    st.pyplot(fig2)
    '''Using Ward's centroid linkage clustering method, countries are effectively grouped into discrete clusters according to the designated indicators, exposing interesting trends in their energy and economic profiles. These clusters give analysts and politicians the ability to create tailored policies, customize interventions, and pinpoint certain issues that are exclusive to each group. A useful tool for future strategic activities and well-informed decision-making, the dendrogram offers a succinct and straightforward picture of the connections and commonalities between nations.'''

    # Evaluation of the algorithm
    sil1 = silhouette_score(scaled_data, Agg1.labels_)  # the silhoutte function takes in the dataset and the lable
    st.write("The evaluation score is", sil1 * 100)

    # Divisive
    div = BisectingKMeans(n_clusters=n_clusters)
    div.fit(scaled_data)
    data['Div Cluster'] = div.labels_
    st.write(data)


def page6():
    st.header("K-Means Clustering")
    cluster_options = [2, 3, 4, 5, 6]

    # create a selectbox
    n_clusters = st.selectbox("Select your preferred number of clusters", cluster_options)

    # Kmeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=30)
    kmeans.fit(scaled_data)

    # diplaying the output. A new column has been added to the origainal data indicating the clusters.
    st.write("Check the last column of this table for the clusters: ")
    scaled_data['Kmeans_clusters'] = kmeans.labels_
    st.write(scaled_data)

  #Evaluation of the algorithm
    sil=silhouette_score(scaled_data,kmeans.labels_) # the silhoutte function takes in the dataset and the label
    st.write("The evaluation score is",sil*100)

    '''K-Means and Ward's Linkage clustering techniques are compared to show their unique advantages and disadvantages. K-Means works well for simple segmentation problems since it produces distinct, non-overlapping clusters with typical mean profiles. Ward's Linkage, on the other hand, provides a thorough hierarchical framework that allows for granular analysis and the identification of natural groupings at different degrees of depth. Ward's Linkage offers flexibility in grouping levels and insight into hierarchical linkages, while K-Means is computationally faster and more scalable. The analytical objectives and dataset properties influence the method selection. K-Means is useful for basic segmentation, but Ward's Linkage is better for a deeper comprehension of hierarchical linkages and groupings. By choosing the right technique, analysts can capitalize on each approach's advantages to gain insightful information from their data.'''


pages = {
    'Group Members': groupC_members,
    'Project Description':page1,
    'Section_A Que.1':section_a_q1,
    'Section_A Que.2':section_a_q2,
    'Section_A Que.3':section_a_q3,
    'Section_A Que.4':section_a_q4,
    'Section_A Que.5':section_a_q5,
    'Dataset': page2,
    'Exploratory Data Analysis': page3,
    'Variable Description': page4,
    'Hierachical Clustering': page5,
    'Kmeans Clustering': page6
}

select_page = st.sidebar.selectbox("select a page", list(pages.keys()))
st.sidebar.header(f'European Renewable Energy Project')
st.sidebar.write(
    'The growing urgency to reduce carbon emissions and their increasingly irreversible effects on global warming has made the transition to renewable energies even more necessary,with regard to electricity generation and the overall energy consumption')

# show the page

pages[select_page]()

