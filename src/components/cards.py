import streamlit as st


def metric_card(
    title,
    value,
    icon="📊",
    description=None,
    container=None,
):
    """Display a metric using native Streamlit components."""

    target = container if container is not None else st

    with target.container(border=True):
        st.markdown(f"**{icon} {title}**")

        st.metric(
            label="Value",
            value=value,
            label_visibility="collapsed",
        )

        if description:
            st.caption(description)


def feature_card(
    title,
    description,
    icon="✨",
):
    """Display a feature card without raw HTML."""

    with st.container(border=True):
        st.markdown(f"### {icon}")

        st.markdown(f"**{title}**")

        st.write(description)


def prediction_result_card(
    predicted_class,
    normal_score,
    pneumonia_score,
):
    """Display the classification result using native components."""

    predicted_class = str(predicted_class).strip().upper()

    st.subheader("Analysis Result")

    if predicted_class == "NORMAL":
        st.success("Model classification: NORMAL")
    elif predicted_class == "PNEUMONIA":
        st.warning("Model classification: PNEUMONIA")
    else:
        st.info(f"Model classification: {predicted_class}")

    col1, col2 = st.columns(2)

    with col1:
        metric_card(
            title="NORMAL output score",
            value=f"{float(normal_score):.4f}",
            icon="🟢",
            description="Output associated with the NORMAL class.",
        )

    with col2:
        metric_card(
            title="PNEUMONIA output score",
            value=f"{float(pneumonia_score):.4f}",
            icon="🟠",
            description="Output associated with the PNEUMONIA class.",
        )

    st.caption(
        "These are model output scores, not necessarily calibrated "
        "probabilities. They must not be interpreted as a diagnosis."
    )