import random
from tempfile import NamedTemporaryFile
from typing import Literal, TypedDict
import streamlit as st

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pydantic import BaseModel, Field
import plotly.express as px
import plotly.graph_objects as go

from datachat.agents.helpers import get_llm

llm = get_llm()


class VizInput(BaseModel):
    question: str = Field(description="Input user question")
    columns: dict = Field(
        description="Dataframe columns and their datatype dict with column name as the key"
    )
    df_path: str = Field(description="Dataframe path where the result set is stored")


@tool(args_schema=VizInput)
def create_chart(question: str, columns: dict, df_path: str):
    """Generates a Chart Type, X and Y labels from the given columns and make it into a matplotlib figure based on user question, columns and df path from langgraph state object.

    Args:
        question: The user question
        columns: Columns available for creating the chart
        df_path: Dataframe path where the result set is stored for creating the chart

    Returns:
        Path where the chart image is stored
    """
    plot_types = ["line", "bar", "scatter", "pie", "heatmap","area","treemap"]
    random.shuffle(plot_types)
    print("???????????????????????????+++++++++++++")
    print(columns)
    #columns = columns.values()
    column_values = []
    for key, value in columns.items():
        if isinstance(value, list):
            column_values.extend(value)
        else:
            column_values.append(value)

    class Graph(TypedDict):
        chart_type: Literal[*plot_types]
        xlabel: Literal[*column_values]
        ylabel: Literal[*column_values]

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are a Data Vizualization expert. "
    #             "Your task is to choose the type of chart ONLY from the given chart types. "
    #             "You should also select what constitutes the x and y axis labels of the chart ONLY from the given columns.\n"
    #             "You'll be given a question for which the user is trying to visualize the insight. "
    #             "You'll also be given a list of columns you can use in creating the chart.\n"
    #             "{format_instructions}"
    #             "Instructions:\n"
    #             "\t1. Donot give the x and y alphabets alone for labels. Choose the x and y labels only from the given column names"
    #             "\t2. Return ONLY the output json. Do not reason the output."
    #             "\t3. The output json should be parseable by Python's json.loads function.",
    #         ),
    #         (
    #             "user",
    #             f"Question: {question}"
    #             f"Columns: {', '.join(columns)}"
    #             f"Chart types (Choose only one that is appropriate for the user question): {plot_types}",
    #         ),
    #     ]
    # )
    prompt = [
        SystemMessage(
            content="You are a Data Vizualization expert. "
            "Your task is to choose the type of chart ONLY from the given chart types. "
            "You should also select what constitutes the x and y axis labels of the chart ONLY from the given columns.\n"
            "You'll be given a question for which the user is trying to visualize the insight. "
            "You'll also be given a list of columns you can use in creating the chart.\n"
            "Instructions:\n"
            "\t1. Donot give the x and y alphabets alone for labels. Choose the x and y labels only from the given column names if there is time column keep it in x column always"
            "\t2. Return ONLY the output json. Do not reason the output."
            "\t3. The output json should be parseable by Python's json.loads function."
            "\t4. Return ONLY the output json which have chart_type, xlabel, ylabel. don't return full decription, Return only values.",
        ),
        HumanMessage(
            content=f"Question: {question}"
                    f"Columns: {', '.join(column_values)}"
                    f"Chart types (Choose only one that is appropriate for the user question): {plot_types}"
        )
    ]
                
    # parser = PydanticOutputParser(pydantic_object=Graph)

    # chain = prompt | llm | parser

    # viz_labels = chain.invoke({"format_instructions": parser.get_format_instructions()})
    viz_labels = llm.with_structured_output(Graph).invoke(prompt)
    print("Output of choose_plot_dims:", viz_labels)
    op_df = pd.read_pickle(df_path)
    plot_type = viz_labels['chart_type']

    # Extract values from all keys except 'x' and 'y'
    extra_columns = []

    column_names = op_df.columns.tolist()
    for key, value in columns.items():
        if isinstance(value, list):
            extra_columns.extend(value)
        else:
            extra_columns.append(value)
    for i in column_names:
        if i not in extra_columns:
            extra_columns.append(i)
    extra_columns.remove(viz_labels['xlabel'])
    extra_columns.remove(viz_labels['ylabel'])

    st.write(extra_columns)
    if len(extra_columns)>0:
        color_col=extra_columns[0]
    else:
        color_col=None


    x_col = columns.get('x')
    y_col = columns.get('y')



    # Initial filter column to use
    filter_col = extra_columns[0] if extra_columns else None
    if not filter_col:

        if op_df.shape == (1, 1):
            # Extract value and label
            value = op_df.iloc[0, 0]
            label = op_df.columns[0]

            # Create an indicator plot
            fig = go.Figure(go.Indicator(
                mode="number",
                value=value,
                title={"text": label}
            ))
        elif op_df.shape[0] < 2 and plot_type in ['pie', 'heatmap']:
            raise ValueError("Not enough data points to plot a meaningful pie or heatmap.")
        elif plot_type == 'scatter':
            fig = px.scatter(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'line':
            fig = px.line(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'bar':
            fig = px.bar(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col, barmode="group")
        elif plot_type == 'pie':
            fig = px.pie(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'heatmap':
            fig = px.density_heatmap(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'treemap':
            fig = px.treemap(
                op_df,
                path=[viz_labels['xlabel']],     # e.g., ['Product']
                values=viz_labels['ylabel'],     # e.g., 'Sales'
                color=color_col                  # optional color dimension
            )
        elif plot_type == 'area':
            fig = px.area(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")


        
        # Customize the layout (like the legend position and size)
        fig.update_layout(
            legend=dict(
                x=1.04,  # This moves the legend outside
                y=1,
                font=dict(size=8),  # Adjusts font size (Plotly uses numeric size values, not 'xx-small')
            )
        )
        return fig
    else:
        print("Hii")
        plot_data = {
            'x': viz_labels['xlabel'],
            'y': viz_labels['ylabel'],
            'color': color_col,
            'filter_col':extra_columns,
            'plot_type':plot_type

        }
        st.session_state.plot=plot_data
        if op_df.shape == (1, 1):
            # Extract value and label
            value = op_df.iloc[0, 0]
            label = op_df.columns[0]

            # Create an indicator plot
            fig = go.Figure(go.Indicator(
                mode="number",
                value=value,
                title={"text": label}
            ))
        elif op_df.shape[0] < 2 and plot_type in ['pie', 'heatmap']:
            raise ValueError("Not enough data points to plot a meaningful pie or heatmap.")
        elif plot_type == 'scatter':
            fig = px.scatter(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'line':
            fig = px.line(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'area':
            fig = px.area(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'bar':
            fig = px.bar(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col, barmode="group")
        elif plot_type == 'pie':
            fig = px.pie(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'heatmap':
            fig = px.density_heatmap(op_df, x=viz_labels['xlabel'], y=viz_labels['ylabel'], color=color_col)
        elif plot_type == 'treemap':
            fig = px.treemap(
                op_df,
                path=[viz_labels['xlabel']],     # e.g., ['Product']
                values=viz_labels['ylabel'],     # e.g., 'Sales'
                color=color_col                  # optional color dimension
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        
        # Customize the layout (like the legend position and size)
        fig.update_layout(
            legend=dict(
                x=1.04,  # This moves the legend outside
                y=1,
                font=dict(size=8),  # Adjusts font size (Plotly uses numeric size values, not 'xx-small')
            )
        )
        return fig

    """
    else:
        # Store traces and visibility maps
        trace_map = {}  # (col, val) -> trace index

        traces = []
        for col in extra_columns:
            values = sorted(op_df[col].dropna().unique())
            for val in values:
                filtered_df = op_df[op_df[col] == val]
                trace_name = f"{col}: {val}"
                if plot_type == 'scatter':
                    trace = go.Scatter(x=filtered_df[x_col], y=filtered_df[y_col], mode='markers', name=trace_name)
                elif plot_type == 'line':
                    trace = go.Scatter(x=filtered_df[x_col], y=filtered_df[y_col], mode='lines+markers', name=trace_name)
                elif plot_type == 'bar':
                    trace = go.Bar(x=filtered_df[x_col], y=filtered_df[y_col], name=trace_name)
                traces.append(trace)
                trace_map[(col, val)] = len(traces) - 1

        fig = go.Figure(data=traces)

        # Add one dropdown menu per column
        updatemenus = []
        for i, col in enumerate(extra_columns):
            values = sorted(op_df[col].dropna().unique())
            buttons = []
            for val in values:
                visibility = [False] * len(traces)
                for (c, v), idx in trace_map.items():
                    if c == col and v == val:
                        visibility[idx] = True
                buttons.append(dict(label=str(val), method="update", args=[{"visible": visibility}, {"title": f"{col} = {val}"}]))

            updatemenus.append(
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.05 + i * 0.2,  # Spread dropdowns horizontally
                    y=1.05,  # Keep them at the top
                    xanchor="left",
                    yanchor="bottom"
                )
            )

        fig.update_layout(
            updatemenus=updatemenus,
            title=f"{plot_type.title()} Chart with Multi-Filter Support",
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend=dict(x=1.05, y=1, font=dict(size=8))
        )
        return fig """

    # fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # op_df.plot(
    #     kind=plot_type,
    #     backend="matplotlib",
    #     x=viz_labels['xlabel'],
    #     y=viz_labels['ylabel'],
    #     ax=ax,
    # ).legend(loc="upper left", bbox_to_anchor=(1.04, 1), fontsize="xx-small")

    # ax.set_xlabel(viz_labels['xlabel'])
    # ax.set_ylabel(viz_labels['ylabel'])
    # fig.subplots_adjust(right=0.7)
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

    # plt.xticks(rotation=45, ha="right")

    # with NamedTemporaryFile(delete=False, suffix=".png") as fig_temp_file:
    #     fig.savefig(
    #         fig_temp_file.name, format="png", bbox_inches="tight", pad_inches=0.1
    #     )  # Do not close the file right away. This file is needed to read the figure in the chat UI
    #     plt.close(fig)
    # return fig_temp_file.name
