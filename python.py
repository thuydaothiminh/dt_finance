# --- Chức năng 6: Khung Chat với Gemini ---
st.subheader("6. Hỏi đáp trực tiếp với Gemini 💬")

# Lưu lịch sử hội thoại vào session_state để không bị reset sau mỗi lần gửi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Giao diện nhập câu hỏi
user_input = st.text_input("Nhập câu hỏi của bạn về tài chính, đầu tư hoặc AI:")

if st.button("Gửi câu hỏi đến Gemini"):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng cấu hình trong Streamlit Secrets.")
    elif not user_input.strip():
        st.warning("⚠️ Vui lòng nhập nội dung trước khi gửi.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            model_name = "gemini-2.5-flash"

            # Gửi nội dung hội thoại trước đó + câu hỏi mới
            chat_prompt = "\n".join(
                [f"Người dùng: {msg['user']}\nGemini: {msg['ai']}" for msg in st.session_state.chat_history]
            )
            full_prompt = f"""
            Bạn là trợ lý AI tài chính chuyên nghiệp của Agribank. 
            Trả lời ngắn gọn, dễ hiểu, có dẫn chứng hoặc ví dụ nếu cần.
            Dưới đây là hội thoại trước đó (nếu có):
            {chat_prompt}

            Câu hỏi mới: {user_input}
            """

            with st.spinner("🤖 Gemini đang phản hồi..."):
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt
                )
                answer = response.text

            # Lưu lịch sử
            st.session_state.chat_history.append({"user": user_input, "ai": answer})

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi gọi Gemini API: {e}")

# Hiển thị lịch sử hội thoại
if st.session_state.chat_history:
    st.write("### 💬 Lịch sử hội thoại")
    for i, msg in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**🧑 Người dùng:** {msg['user']}")
        st.markdown(f"**🤖 Gemini:** {msg['ai']}")
        st.markdown("---")
