# python.py
import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Financial Report Analysis App",
    layout="wide"
)
st.title("Financial Report Analysis Application 📊")

# --- Main Calculation Function (Uses Caching for Performance) ---
@st.cache_data
def process_financial_data(df):
    """Calculates Growth and Ratios."""
    # Ensure values are numeric for calculations
    numeric_cols = ['Previous Year', 'Next Year']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Calculate Growth Rate
    # Use .replace(0, 1e-9) for Pandas Series to avoid division by zero errors
    df['Growth Rate (%)'] = (
        (df['Next Year'] - df['Previous Year']) / df['Previous Year'].replace(0, 1e-9)
    ) * 100

    # 2. Calculate Weight by Total Assets
    # Filter for the "TOTAL ASSETS" indicator
    total_assets_row = df[df['Indicator'].str.contains('TOTAL ASSETS', case=False, na=False)]
    if total_assets_row.empty:
        raise ValueError("The 'TOTAL ASSETS' indicator was not found.")
    
    total_assets_N_1 = total_assets_row['Previous Year'].iloc[0]
    total_assets_N = total_assets_row['Next Year'].iloc[0]

    # ******************************* BEGIN ERROR FIX *******************************
    # Errors occur when using .replace() on single values (numpy.int64).
    # Use ternary conditions to manually handle zero values for the denominator.
    divisor_N_1 = total_assets_N_1 if total_assets_N_1 != 0 else 1e-9
    divisor_N = total_assets_N if total_assets_N != 0 else 1e-9

    # Calculate the weight with the denominator processed
    df['Weight Previous Year (%)'] = (df['Previous Year'] / divisor_N_1) * 100
    df['Weight Next Year (%)'] = (df['Next Year'] / divisor_N) * 100
    # ******************************* END ERROR FIX *******************************

    return df

# --- Gemini API Call Function ---
def get_ai_analysis(data_for_ai, api_key):
    """Sends analysis data to the Gemini API and receives comments."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        prompt = f"""
        You are a professional financial analyst. Based on the following financial indicators, provide an objective and concise comment (about 3-4 paragraphs) on the company's financial situation. Focus on growth rate, changes in asset structure, and current solvency.
        Raw data and indicators:
        {data_for_ai}
        """
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Gemini API call error: Please check your API Key or usage limits. Error details: {e}"
    except KeyError:
        return "Error: The 'GEMINI_API_KEY' API key was not found. Please check the Secrets configuration on Streamlit Cloud."
    except Exception as e:
        return f"An unidentified error occurred: {e}"

# --- Gemini API Call Function for Chat ---
def get_chat_response(full_data, user_prompt, api_key):
    """Sends all data and user questions to the Gemini API and receives answers."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        prompt = f"""
        You are a professional financial analyst. Based on the following detailed financial data, answer the user's question.
        Detailed data:
        {full_data.to_markdown(index=False)}

        User question: {user_prompt}
        """
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Gemini API call error: {e}"
    except Exception as e:
        return f"Unidentified error: {e}"

# --- Function 1: Upload File ---
uploaded_file = st.file_uploader(
    "1. Upload Excel File Financial Report (Indicator | Previous Year | Next Year)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        # Preprocessing: Ensure only 3 important columns
        df_raw.columns = ['Indicator', 'Previous Year', 'Next Year']
        
        # Process data
        df_processed = process_financial_data(df_raw.copy())
        
        if df_processed is not None:
            # --- Function 2 & 3: Display Results ---
            st.subheader("2. Growth Rate & 3. Asset Structure Weight")
            st.dataframe(df_processed.style.format({
                'Previous Year': '{:,.0f}',
                'Next Year': '{:,.0f}',
                'Growth Rate (%)': '{:.2f}%',
                'Weight Previous Year (%)': '{:.2f}%',
                'Weight Next Year (%)': '{:.2f}%'
            }), use_container_width=True)

            # --- Function 4: Calculate Financial Indicators ---
            st.subheader("4. Basic Financial Indicators")
            try:
                # Filter values for Current Liquidity Ratio (Example)
                short_term_assets_n = df_processed[df_processed['Indicator'].str.contains('SHORT-TERM ASSETS', case=False, na=False)]['Next Year'].iloc[0]
                short_term_assets_n_1 = df_processed[df_processed['Indicator'].str.contains('SHORT-TERM ASSETS', case=False, na=False)]['Previous Year'].iloc[0]
                
                short_term_debt_N = df_processed[df_processed['Indicator'].str.contains('SHORT-TERM DEBT', case=False, na=False)]['Next Year'].iloc[0]
                short_term_debt_N_1 = df_processed[df_processed['Indicator'].str.contains('SHORT-TERM DEBT', case=False, na=False)]['Previous Year'].iloc[0]

                # Calculation
                current_liquidity_N = short_term_assets_n / short_term_debt_N
                current_liquidity_N_1 = short_term_assets_n_1 / short_term_debt_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Current Liquidity Ratio (Previous Year)",
                        value=f"{current_liquidity_N_1:.2f} times"
                    )
                with col2:
                    st.metric(
                        label="Current Liquidity Ratio (Next Year)",
                        value=f"{current_liquidity_N:.2f} times",
                        delta=f"{current_liquidity_N - current_liquidity_N_1:.2f}"
                    )
            except IndexError:
                st.warning("The 'SHORT-TERM ASSETS' or 'SHORT-TERM DEBT' indicators are missing to calculate the index.")
                current_liquidity_N = "N/A"
                current_liquidity_N_1 = "N/A"
            except ZeroDivisionError:
                st.warning("Cannot calculate current liquidity ratio because Short-Term Debt is 0.")
                current_liquidity_N = "N/A"
                current_liquidity_N_1 = "N/A"

            # --- Function 5: AI Comments (initial) ---
            st.subheader("5. Financial Situation Comments (AI)")
            data_for_ai = pd.DataFrame({
                'Indicator': [
                    'Entire Analysis Table (raw data)',
                    'Short-term asset growth (%)',
                    'Current liquidity (N-1)',
                    'Current liquidity (N)'
                ],
                'Value': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Indicator'].str.contains('SHORT-TERM ASSETS', case=False, na=False)]['Growth Rate (%)'].iloc[0]:.2f}%" if 'SHORT-TERM ASSETS' in df_processed['Indicator'].to_list() else "N/A",
                    f"{current_liquidity_N_1}",
                    f"{current_liquidity_N}"
                ]
            }).to_markdown(index=False)
            
            if st.button("Request AI Analysis"):
                api_key = st.secrets.get("GEMINI_API_KEY")
                if api_key:
                    with st.spinner('Sending data and waiting for Gemini to analyze...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                    st.markdown("**Analysis Results from Gemini AI:**")
                    st.info(ai_result)
                else:
                    st.error("Error: API Key not found. Please configure the 'GEMINI_API_KEY' key in Streamlit Secrets.")

            # --- Function 6: AI Chat Frame ---
            st.subheader("6. AI Chat about Financial Reports")

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display previous messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Process user input
            if prompt := st.chat_input("Ask about the financial report..."):
                # Add the user's message to the history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Prepare data for AI
                full_data_string = df_processed.to_markdown(index=False)
                
                # Call Gemini API and receive a response
                api_key = st.secrets.get("GEMINI_API_KEY")
                if api_key:
                    with st.chat_message("assistant"):
                        with st.spinner("Sending question and waiting for Gemini to respond..."):
                            response = get_chat_response(df_processed, prompt, api_key)
                            st.markdown(response)
                            # Add AI's response to the history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Error: API Key not found. Please configure the 'GEMINI_API_KEY' key in Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Data structure error: {ve}")
    except Exception as e:
        st.error(f"An error occurred while reading or processing the file: {e}. Please check the file format.")
else:
    st.info("Please upload the Excel file to begin the analysis.")
