import plotly.graph_objects as go

def plot_bar_chart(filtered_df, selected_stock_symbol):
    if not filtered_df.empty and 'sym' in filtered_df.columns and 'ps' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='ps', ascending=False)
        
        fig_bar = go.Figure()

        # Add bar chart
        fig_bar.add_trace(go.Bar(
            x=filtered_df['sym'], y=filtered_df['ps'],
            marker_color='steelblue', name='PS Metric',
            text=filtered_df['ps'],  # Add data labels
            textposition='auto',
            hoverinfo='text+name',
            hovertemplate='<b>Symbol:</b> %{x}<br><b>PS Metric:</b> %{y}<extra></extra>'
        ))

        # Highlight selected symbol
        fig_bar.update_traces(
            marker=dict(color=['orange' if sym == selected_stock_symbol else 'dodgerblue' for sym in filtered_df['sym']])
        )

        # Customize layout
        fig_bar.update_layout(
            #title="PS Metric Bar Chart",
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis={  
                'fixedrange':True, 
                'ticklabelposition': 'inside', 
                'tickfont': {'size': 12, 'color': 'LightSteelBlue'},
                'tickcolor': 'LightSteelBlue',
                },
            yaxis={
                'fixedrange':True, 
                'ticklabelposition': 'inside', 
                'tickfont': {'size': 12, 'color': 'LightSteelBlue'},
                'tickcolor': 'LightSteelBlue',
                },
            modebar=dict(remove=["zoom", "pan", "select2d", "lasso2d", "autoScale", "resetScale", "zoomIn", "zoomOut", "resetViews"])
        )

        return fig_bar
    else:
        return None