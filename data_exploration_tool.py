import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# Function to upload and load the dataset
def load_data():
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Print data types for debugging
        st.write("Data Types:", df.dtypes)

        return df
    return None

# Function to display basic statistics
def display_statistics(df):
    st.subheader("Basic Data Summary Statistics")

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numerical_cols:
        st.write("**Numerical Columns Summary**")
        st.write(df[numerical_cols].describe())
    else:
        st.write("No numerical columns found.")

    if categorical_cols:
        st.write("**Categorical Columns Summary**")
        st.write(df[categorical_cols].describe(include='all'))
    else:
        st.write("No categorical columns found.")


def missing_data_analysis(df):
    st.subheader("Missing Data Analysis")

    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_data_summary = missing_values[missing_values > 0]

    # Show percentage of missing values
    missing_percentage = (missing_values / df.shape[0]) * 100
    missing_percentage_summary = missing_percentage[missing_values > 0]

    if not missing_data_summary.empty:
        st.write("**Columns with Missing Values**")
        st.write(missing_data_summary)
        st.write("**Percentage of Missing Values**")
        st.write(missing_percentage_summary)

        # # Visualization of missing data without explicit type conversion
        # try:
        #     plt.figure(figsize=(10, 6))
        #     sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        #     st.pyplot(plt)
        # except Exception as e:
        #     st.warning("Unable to visualize missing data: " + str(e))

        # Check for completely blank columns
        blank_columns = df.columns[df.isnull().all()].tolist()
        if blank_columns:
            st.write("**Completely Blank Columns**")
            st.write(blank_columns)

            # Option to delete blank columns
            col_to_delete = st.selectbox("Select a column to delete:", blank_columns + ["None"])
            if col_to_delete != "None":
                df.drop(col_to_delete, axis=1, inplace=True)
                st.success(f"Column '{col_to_delete}' has been deleted.")

        # Options for handling missing values
        st.subheader("Options for Handling Missing Values")
        impute_method = st.selectbox("Select imputation method:", ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value", "None"])

        if impute_method != "None":
            if impute_method == "Mean":
                for col in df.select_dtypes(include=['float64', 'int64']).columns:
                    df[col].fillna(df[col].mean(), inplace=True)
                st.success("Missing values filled with mean.")
                
            elif impute_method == "Median":
                for col in df.select_dtypes(include=['float64', 'int64']).columns:
                    df[col].fillna(df[col].median(), inplace=True)
                st.success("Missing values filled with median.")
                
            elif impute_method == "Mode":
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                st.success("Missing values filled with mode.")

            elif impute_method == "Forward Fill":
                df.fillna(method='ffill', inplace=True)
                st.success("Missing values filled using forward fill method.")

            elif impute_method == "Backward Fill":
                df.fillna(method='bfill', inplace=True)
                st.success("Missing values filled using backward fill method.")

            elif impute_method == "Custom Value":
                custom_value = st.text_input("Enter custom value:")
                if st.button("Fill with Custom Value"):
                    for col in df.columns:
                        df[col].fillna(custom_value, inplace=True)
                    st.success(f"Missing values filled with custom value: {custom_value}")

        # Option to remove rows with any missing values
        if st.button("Remove Rows with Missing Values"):
            original_row_count = df.shape[0]
            df.dropna(inplace=True)
            st.success(f"Removed {original_row_count - df.shape[0]} rows with missing values.")

    else:
        st.write("No missing values found.")


def clean_data(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        st.subheader("Data Cleaning Options")
        
        # Handle missing values
        if st.checkbox("Drop rows with missing values"):
            original_shape = df.shape[0]
            df = df.dropna()
            st.write(f"Dropped {original_shape - df.shape[0]} rows with missing values.")
        
        # Filter rows
        if st.checkbox("Filter rows by condition"):
            columns = df.columns.tolist()
            column = st.selectbox("Select the column to filter by", columns)
            unique_values = df[column].unique().tolist()
            filter_value = st.selectbox("Select the value to filter rows", unique_values)
            df = df[df[column] == filter_value]
            st.write(f"Filtered rows where {column} is {filter_value}.")
    else:
        st.write("No missing values found. No cleaning options available.")
    
    return df


# Function to visualize data
def visualize_data(df):
    st.subheader("Interactive Data Visualizations")

    # Select chart type
    chart_type = st.selectbox("Select the type of chart", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Area Chart", "Heatmap"])

    # Chart Selection
    if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Area Chart"]:
        x_axis = st.selectbox("Select X-axis", df.columns)
        
        # Determine possible Y-axis options
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if chart_type in ["Bar Chart", "Line Chart", "Area Chart"]:
            if df[x_axis].dtype == 'object':
                st.warning("For Bar/Line/Area Charts, the X-axis is usually a categorical variable. Ensure the data is categorical.")
        
        y_axis = st.selectbox("Select Y-axis", numeric_cols)
        
        # Select aggregation method for Y-axis
        aggregation_method = st.selectbox("Select aggregation method", ["Mean", "Sum"])
        
        if aggregation_method == "Mean":
            df_grouped = df.groupby(x_axis).mean().reset_index()
        else:
            df_grouped = df.groupby(x_axis).sum().reset_index()

        if chart_type == "Bar Chart":
            fig = px.bar(df_grouped, x=x_axis, y=y_axis, title=f"Bar Chart of {y_axis} by {x_axis}")
            st.plotly_chart(fig)

        elif chart_type == "Line Chart":
            fig = px.line(df_grouped, x=x_axis, y=y_axis, title=f"Line Chart of {y_axis} by {x_axis}")
            st.plotly_chart(fig)

        elif chart_type == "Area Chart":
            fig = px.area(df_grouped, x=x_axis, y=y_axis, title=f"Area Chart of {y_axis} by {x_axis}")
            st.plotly_chart(fig)

        # Scatter Plot
        if chart_type == "Scatter Plot":
            color = st.selectbox("Select color", df.columns)
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f"Scatter Plot of {y_axis} vs {x_axis}")
            st.plotly_chart(fig)

    # Pie Chart
    elif chart_type == "Pie Chart":
        if df.select_dtypes(include=['object']).shape[1] > 0:
            pie_column = st.selectbox("Select column for Pie Chart", df.select_dtypes(include=['object']).columns)
            pie_data = df[pie_column].value_counts().reset_index()
            fig = px.pie(pie_data, names='index', values=pie_column, title=f"Pie Chart of {pie_column}")
            st.plotly_chart(fig)
        else:
            st.warning("No categorical columns available for pie chart.")

    # Heatmap
    elif chart_type == "Heatmap":
        if df.select_dtypes(include=[float, int]).shape[1] > 1:
            corr_matrix = df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Heatmap of Correlation Matrix")
            st.plotly_chart(fig)
        else:
            st.warning("Heatmap requires at least two numerical columns.")


# Main function to run the Streamlit app
def main():
    st.title("Interactive Data Exploration Tool")

    # Load data
    df = load_data()

    if df is not None:
        st.subheader("Dataset Overview")
        
        # Display the number of rows and columns
        st.write(f"**Number of rows:** {df.shape[0]}")
        st.write(f"**Number of columns:** {df.shape[1]}")
        
        st.write("First 5 rows of the dataset:")
        st.dataframe(df.head())

        # Display statistics
        display_statistics(df)
        
        # Missing data analysis
        missing_data_analysis(df)
        
        # Clean data
        df = clean_data(df)
        
        # Visualize data
        visualize_data(df)
        print("**********")
        print("Improvements for this app are currently in progress. Exciting new features are coming soon!")


if __name__ == "__main__":
    main()
    
