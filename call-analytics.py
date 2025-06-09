# streamlit_app.py

import streamlit as st
import pandas as pd
import json
import requests
import base64
import traceback
import copy
import io
import hashlib

st.title("Cohort Creation with Sarvam")

# --- Authentication using st.secrets ---
USER_CREDENTIALS = st.secrets["USER_CREDENTIALS"]

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

with st.sidebar:
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

if login_button:
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.session_state["authenticated"] = True
        st.success(f"Welcome, {username}!")
    else:
        st.error("Invalid username or password")

if not st.session_state.get("authenticated", False):
    st.stop()


# --- 2. Upload CSV & Select Calls ---
st.header("1. Upload Interaction CSV")

# Add user inputs for number of top and average duration calls
col1, col2 = st.columns(2)
with col1:
    num_top = st.number_input(
        "Number of highest duration calls to analyze",
        min_value=0,
        value=25,
        step=1,
        format="%d"
    )
with col2:
    num_avg = st.number_input(
        "Number of average duration calls to analyze",
        min_value=0,
        value=25,
        step=1,
        format="%d"
    )

interaction_csv = st.file_uploader("Upload interactions CSV", type=["csv"])

if interaction_csv:
    df = pd.read_csv(interaction_csv)
    if 'Duration' not in df.columns:
        st.error("Missing 'Duration' column in CSV")
    else:
        df_sorted = df.sort_values(by='Duration', ascending=False)
        topN = df_sorted.head(num_top)
        avg_duration = df['Duration'].mean()
        avgN = df.iloc[(df['Duration'] - avg_duration).abs().argsort()[:num_avg]]
        selected_df = pd.concat([topN, avgN]).drop_duplicates()
        st.success(f"Selected {len(selected_df)} transcripts ({num_top} longest + {num_avg} average)")
        st.dataframe(selected_df.head())

# --- 3. Upload or Generate JSON from Steps ---
st.header("2. Upload or Generate Call Steps JSON")

steps_source = st.radio("Choose how to provide call steps", ["Upload JSON", "Enter Text"])
questions = []

if steps_source == "Upload JSON":
    json_file = st.file_uploader("Upload call steps JSON", type=["json"])
    if json_file:
        steps_json = json.load(json_file)
        st.session_state["steps_json"] = steps_json
        questions = steps_json.get("questions", [])
    elif "steps_json" in st.session_state:
        steps_json = st.session_state["steps_json"]
        questions = steps_json.get("questions", [])
else:
    text_steps = st.text_area("Enter call steps in text")
    if text_steps and st.button("Generate JSON via GPT"):
        try:
            BASE_URL = "https://azure-openai-deployment-agents.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
            HEADERS = {
                "Content-Type": "application/json",
                "api-key": st.secrets["USER_CREDENTIALS"]["azure_openai_key"]
            }
            response = requests.post(
                BASE_URL,
                headers=HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert assistant helping convert call steps into structured compliance-oriented JSON."},
                        {"role": "user", "content": f"Convert the following call steps into a JSON object with a 'questions' key. Each question should have an string 'id' like q1, q2 etc., 'text', and 'type' = 'boolean'. Ensure the phrasing is formal and compliance-focused. Steps: {text_steps}"}
                    ],
                    "temperature": 0.2
                }
            )
            if response.status_code != 200:
                st.error(f"GPT API call failed: {response.status_code}")
                st.text(response.text)
            else:
                try:
                    content = response.json()
                    st.code(json.dumps(content, indent=2), language="json")
                    model_output = content.get("choices", [{}])[0].get("message", {}).get("content", "").strip('`json').strip('`').strip()
                    steps_json = json.loads(model_output)
                    questions = steps_json.get("questions", [])
                    st.session_state["steps_json"] = steps_json  # Persist in session
                    st.session_state["questions"] = questions   # Persist questions in session
                    st.download_button("Download JSON", json.dumps(steps_json, indent=2), file_name="steps.json")
                except Exception as parse_err:
                    st.error("Failed to parse GPT output.")
                    st.text(f"Raw response: {response.text}")
                    st.text(traceback.format_exc())
        except Exception as e:
            st.error("Error calling GPT step-parser API")
            st.text(traceback.format_exc())
    # If questions are in session_state, use them
    if "questions" in st.session_state:
        questions = st.session_state["questions"]
    # Add note below the button
    st.info("This requires use of GPT credits. Kindly download the steps once created and then upload for future use.")

# --- Add ASR and Disposition Validation Options ---
st.markdown("---")
st.subheader("Optional Automated Checks")
add_asr = st.checkbox("Check ASR performance (add ASR validation question)")
add_dis = st.checkbox("Validate dispositions (add disposition validation question)")

# Define the ASR and Disposition questions
asr_question = {
    "id": "q_asr",
    "text": "Does the transcript contain any signs of speech recognition errors (ASR issues), such as nonsensical phrases, incoherent language, or mistranscriptions that distort meaning?",
    "type": "boolean"
}
dis_question = {
    "id": "q_dis",
    "text": "Does the final disposition match the conversation? (e.g., if the user agrees to make a payment, the disposition should be 'promise_to_pay'; if the user refuses, it should be 'refused_to_pay'). Use the provided disposition variable for this check.",
    "type": "boolean"
}

# Add selected questions to the end of the list and update steps_json
if add_asr and (not any(q.get('id') == 'q_asr' for q in questions)):
    questions.append(asr_question)
if add_dis and (not any(q.get('id') == 'q_dis' for q in questions)):
    questions.append(dis_question)

# Also update steps_json in session_state if it exists
if (add_asr or add_dis) and "steps_json" in st.session_state:
    st.session_state["steps_json"]["questions"] = questions

# --- 4. Display Questions Box ---
if questions:
    st.sidebar.header("Evaluation Questions (from JSON)")
    for q in questions:
        st.sidebar.markdown(f"**{q['id']}**: {q['text']}")

# --- 5. Run Analytics ---
st.header("3. Run Analytics")
status_box = st.empty()

# Take Sarvam API key as input from user
sarvam_api_key = st.text_input("Enter your Sarvam API Key", type="password")

# Add a stop button and state
if "stop_analytics" not in st.session_state:
    st.session_state["stop_analytics"] = False
if "analytics_running" not in st.session_state:
    st.session_state["analytics_running"] = False

run_col, stop_col = st.columns([3, 1])
with run_col:
    if st.button("Run Sarvam Text Analytics", key="run_analytics_btn"):
        st.session_state["stop_analytics"] = False  # Reset stop flag at start
        st.session_state["analytics_running"] = True
        if not sarvam_api_key:
            st.error("Please enter your Sarvam API key.")
            st.session_state["analytics_running"] = False
        else:
            try:
                results = []
                responses = [] # Store API responses
                total = len(selected_df)
                done = 0
                failed = 0
                for idx, row in selected_df.iterrows():
                    if st.session_state.get("stop_analytics", False):
                        st.warning("Analytics stopped by user. Outputting results so far.")
                        break
                    interaction_id = row['Interaction_ID'] if 'Interaction_ID' in row else f'Row {idx+1}'
                    pending = total - (done + failed)
                    status_box.info(f"Analyzing Interaction ID: {interaction_id} | Done: {done} | Failed: {failed} | Pending: {pending}")
                    try:
                        # Prepare questions for this row, injecting disposition if needed
                        row_questions = copy.deepcopy(questions)
                        for q in row_questions:
                            if q['id'] == 'q_dis':
                                q['disposition'] = row.get('Disposition', '')
                        payload = {
                            "text": row['Transcript'],
                            "questions": json.dumps(row_questions)
                        }
                        res = requests.post(
                            "https://api.sarvam.ai/text-analytics",
                            headers={
                                "api-subscription-key": sarvam_api_key,
                                "Content-Type": "application/x-www-form-urlencoded"
                            },
                            data=payload
                        )
                        try:
                            response_json = res.json()
                        except Exception:
                            response_json = {"error": res.text}
                        responses.append(json.dumps({"interaction_id": interaction_id, "response": response_json}, ensure_ascii=False))
                        if res.status_code != 200:
                            raise Exception(f"API error for Interaction {interaction_id}: {res.text}")
                        ans = response_json.get('answers', [])
                        # Build both simple and detailed records
                        record_simple = {f"answer_{a['id']}": (a.get('response', 'N/A')) for a in ans}
                        record_detailed = {}
                        for a in ans:
                            qid = a['id']
                            record_detailed[f"answer_{qid}"] = a.get('response', 'N/A')
                            record_detailed[f"reasoning_{qid}"] = a.get('reasoning', '')
                            record_detailed[f"utterance_{qid}"] = a.get('utterance', '')
                        record_simple['Interaction_ID'] = interaction_id
                        record_simple['Language'] = row.get('Language', 'Unknown')
                        record_detailed['Interaction_ID'] = interaction_id
                        record_detailed['Language'] = row.get('Language', 'Unknown')
                        results.append({'simple': record_simple, 'detailed': record_detailed})
                        done += 1
                        # If all answers are 'No', add a note for troubleshooting
                        if all(v == 'No' for k, v in record_simple.items() if k.startswith('answer_')):
                            st.warning(f"All answers are 'No' for Interaction ID {interaction_id}. Please check your question format and API expectations.")
                    except Exception as api_err:
                        failed += 1
                        st.warning(f"Failed to analyze Interaction {interaction_id}: {api_err}")
                        continue
                # Prepare output DataFrames
                output_df_simple = pd.DataFrame([r['simple'] for r in results])
                output_df_detailed = pd.DataFrame([r['detailed'] for r in results])
                st.session_state["analytics_result_simple"] = output_df_simple
                st.session_state["analytics_result_detailed"] = output_df_detailed
                st.success(f"Analytics completed! Done: {done}, Failed: {failed}, Total: {total}")
                st.session_state["analytics_running"] = False
                # Add toggle for simple/detailed output
                st.session_state["show_detailed"] = False
            except Exception as e:
                st.error("Failed to run analytics")
                status_box.error("Error occurred during analytics execution.")
                st.text(traceback.format_exc())
                st.session_state["analytics_running"] = False
with stop_col:
    if st.session_state.get("analytics_running", False):
        if st.button("Stop Analytics", key="stop_analytics_btn"):
            st.session_state["stop_analytics"] = True

# --- Output Table and Download Buttons (always visible if results exist) ---
if "analytics_result_simple" in st.session_state or "analytics_result_detailed" in st.session_state:
    show_detailed = st.checkbox("Show detailed output (reasoning, utterance)", value=st.session_state.get("show_detailed", False), key="analytics_toggle")
    st.session_state["show_detailed"] = show_detailed
    if show_detailed and "analytics_result_detailed" in st.session_state:
        output_df_detailed = st.session_state["analytics_result_detailed"]
        st.dataframe(output_df_detailed.head())
        st.download_button("Download Detailed Output CSV", output_df_detailed.to_csv(index=False), file_name="analytics_output_detailed.csv", mime="text/csv")
    elif "analytics_result_simple" in st.session_state:
        output_df_simple = st.session_state["analytics_result_simple"]
        st.dataframe(output_df_simple.head())
        st.download_button("Download Output CSV", output_df_simple.to_csv(index=False), file_name="analytics_output.csv", mime="text/csv")

# --- 6. Visualizations ---
st.header("4. Visualize Results")
if "analytics_result_simple" in st.session_state or "analytics_result_detailed" in st.session_state:
    import matplotlib.pyplot as plt
    import seaborn as sns

    show_detailed = st.checkbox("Show detailed output (reasoning, utterance) for visualization", value=False, key="viz_toggle")
    if show_detailed and "analytics_result_detailed" in st.session_state:
        df_result = st.session_state["analytics_result_detailed"]
    else:
        df_result = st.session_state["analytics_result_simple"]

    # Only use question IDs for labels
    question_ids = [col.replace('answer_', '') for col in df_result.columns if col.startswith('answer_')]
    df_numeric = df_result[[f'answer_{qid}' for qid in question_ids]].applymap(lambda x: 1 if x == 'Yes' else 0)
    accuracy = df_numeric.mean() * 100
    labels = question_ids

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(x=accuracy, y=labels, palette="mako", ax=ax)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Question ID")
    ax.set_title("Gen AI Voice Bot: Per-Question Accuracy")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    st.pyplot(fig)

    df_result['compliance_score'] = df_numeric.sum(axis=1)
    fig2, ax2 = plt.subplots()
    sns.histplot(df_result['compliance_score'], bins=range(0, len(labels)+2), kde=False, ax=ax2)
    ax2.set_title("Compliance Score per Interaction")
    ax2.set_xlabel("Number of 'Yes' Responses")
    ax2.set_ylabel("Number of Interactions")
    st.pyplot(fig2)

    # Always show download and preview if analytics_result exists
    st.dataframe(df_result.head())
    st.download_button("Download Output CSV", df_result.to_csv(index=False), file_name="analytics_output.csv", mime="text/csv")
