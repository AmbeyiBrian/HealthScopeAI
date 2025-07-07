"""
Utility functions for the HealthScopeAI Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_number(num: int) -> str:
    """
    Format number with appropriate suffix (K, M, B).
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def calculate_health_score(data: Dict[str, int]) -> float:
    """
    Calculate a health score based on mention counts.
    
    Args:
        data: Dictionary with health condition counts
        
    Returns:
        Health score (0-100)
    """
    total_mentions = data.get('total_health_mentions', 0)
    
    if total_mentions == 0:
        return 0
    
    # Weight different conditions
    weights = {
        'depression': 3,
        'anxiety': 3,
        'stress': 2,
        'flu': 1,
        'fever': 1,
        'pain': 2
    }
    
    weighted_score = 0
    for condition, count in data.items():
        if condition in weights:
            weighted_score += count * weights[condition]
    
    # Normalize to 0-100 scale
    max_possible = total_mentions * 3  # Maximum weight
    return min(100, (weighted_score / max_possible) * 100) if max_possible > 0 else 0

def create_trend_chart(df: pd.DataFrame, location: str = None) -> go.Figure:
    """
    Create a trend chart for health mentions.
    
    Args:
        df: DataFrame with health data
        location: Optional location filter
        
    Returns:
        Plotly figure
    """
    # Filter data if location specified
    if location:
        df = df[df['location'] == location]
    
    # Group by date
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_data = df.groupby('date').agg({
        'is_health_related': 'sum',
        'text': 'count'
    }).reset_index()
    
    daily_data.columns = ['date', 'health_mentions', 'total_posts']
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['health_mentions'],
        mode='lines+markers',
        name='Health Mentions',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_data['date'],
        y=daily_data['total_posts'],
        mode='lines+markers',
        name='Total Posts',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'Health Trends{"" if not location else f" - {location.title()}"}',
        xaxis_title='Date',
        yaxis_title='Health Mentions',
        yaxis2=dict(
            title='Total Posts',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    return fig

def create_condition_heatmap(aggregated_data: Dict[str, Dict[str, int]]) -> go.Figure:
    """
    Create a heatmap showing conditions by location.
    
    Args:
        aggregated_data: Aggregated health data
        
    Returns:
        Plotly figure
    """
    # Extract data for heatmap
    locations = list(aggregated_data.keys())
    conditions = set()
    
    for location_data in aggregated_data.values():
        conditions.update(location_data.keys())
    
    conditions = [c for c in conditions if c != 'total_health_mentions']
    
    # Create matrix
    matrix = []
    for condition in conditions:
        row = []
        for location in locations:
            count = aggregated_data[location].get(condition, 0)
            row.append(count)
        matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[loc.title() for loc in locations],
        y=[cond.replace('_', ' ').title() for cond in conditions],
        colorscale='Reds',
        hovertemplate='<b>%{y}</b><br>%{x}: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Health Conditions by Location',
        xaxis_title='Location',
        yaxis_title='Condition'
    )
    
    return fig

def create_severity_gauge(count: int, max_count: int = 50) -> go.Figure:
    """
    Create a gauge chart for alert severity.
    
    Args:
        count: Current count
        max_count: Maximum expected count
        
    Returns:
        Plotly figure
    """
    # Calculate percentage
    percentage = min(100, (count / max_count) * 100)
    
    # Determine color
    if percentage >= 80:
        color = "red"
    elif percentage >= 50:
        color = "orange"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=count,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Mentions"},
        gauge={
            'axis': {'range': [None, max_count]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_count * 0.3], 'color': "lightgray"},
                {'range': [max_count * 0.3, max_count * 0.7], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_count * 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

def generate_summary_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate summary statistics for the dashboard.
    
    Args:
        df: DataFrame with health data
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_posts': len(df),
        'health_posts': df['is_health_related'].sum() if 'is_health_related' in df.columns else 0,
        'unique_locations': df['location'].nunique(),
        'date_range': {
            'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'end': df['timestamp'].max() if 'timestamp' in df.columns else None
        },
        'top_locations': df['location'].value_counts().head(5).to_dict(),
        'health_ratio': 0
    }
    
    if stats['total_posts'] > 0:
        stats['health_ratio'] = stats['health_posts'] / stats['total_posts']
    
    return stats

def filter_data_by_date(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Filter data by date range.
    
    Args:
        df: DataFrame to filter
        start_date: Start date
        end_date: End date
        
    Returns:
        Filtered DataFrame
    """
    if 'timestamp' not in df.columns:
        return df
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    return df[mask]

def create_location_comparison(aggregated_data: Dict[str, Dict[str, int]], 
                             locations: List[str]) -> go.Figure:
    """
    Create a comparison chart for selected locations.
    
    Args:
        aggregated_data: Aggregated health data
        locations: List of locations to compare
        
    Returns:
        Plotly figure
    """
    # Extract data for comparison
    conditions = set()
    for location in locations:
        if location in aggregated_data:
            conditions.update(aggregated_data[location].keys())
    
    conditions = [c for c in conditions if c != 'total_health_mentions']
    
    # Create grouped bar chart
    fig = go.Figure()
    
    for condition in conditions:
        values = []
        for location in locations:
            count = aggregated_data.get(location, {}).get(condition, 0)
            values.append(count)
        
        fig.add_trace(go.Bar(
            name=condition.replace('_', ' ').title(),
            x=[loc.title() for loc in locations],
            y=values
        ))
    
    fig.update_layout(
        title='Health Conditions Comparison by Location',
        xaxis_title='Location',
        yaxis_title='Count',
        barmode='group'
    )
    
    return fig

def calculate_alert_priority(condition: str, count: int, location: str) -> int:
    """
    Calculate alert priority based on condition, count, and location.
    
    Args:
        condition: Health condition
        count: Number of mentions
        location: Location
        
    Returns:
        Priority score (higher = more urgent)
    """
    # Base priority from count
    priority = count
    
    # Condition weights
    condition_weights = {
        'depression': 3,
        'anxiety': 3,
        'suicide': 5,
        'chest_pain': 4,
        'fever': 2,
        'flu': 1
    }
    
    # Apply condition weight
    priority *= condition_weights.get(condition, 1)
    
    # Location weights (major cities get higher priority)
    location_weights = {
        'nairobi': 2,
        'mombasa': 1.5,
        'kisumu': 1.2,
        'nakuru': 1.1
    }
    
    # Apply location weight
    priority *= location_weights.get(location.lower(), 1)
    
    return int(priority)

def export_data_to_csv(data: Dict[str, Dict[str, int]], filename: str) -> str:
    """
    Export aggregated data to CSV format.
    
    Args:
        data: Aggregated health data
        filename: Output filename
        
    Returns:
        CSV data as string
    """
    # Convert to DataFrame
    rows = []
    for location, conditions in data.items():
        row = {'location': location}
        row.update(conditions)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def generate_health_report(aggregated_data: Dict[str, Dict[str, int]], 
                         timeframe: str = "current") -> str:
    """
    Generate a text health report.
    
    Args:
        aggregated_data: Aggregated health data
        timeframe: Timeframe for the report
        
    Returns:
        Formatted health report
    """
    total_mentions = sum(
        data.get('total_health_mentions', 0) 
        for data in aggregated_data.values()
    )
    
    # Top locations
    top_locations = sorted(
        aggregated_data.items(),
        key=lambda x: x[1].get('total_health_mentions', 0),
        reverse=True
    )[:3]
    
    # Most common conditions
    condition_counts = {}
    for location_data in aggregated_data.values():
        for condition, count in location_data.items():
            if condition != 'total_health_mentions':
                condition_counts[condition] = condition_counts.get(condition, 0) + count
    
    top_conditions = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Generate report
    report = f"""
    # Health Trends Report - {timeframe.title()}
    
    ## Summary
    - Total health mentions: {total_mentions}
    - Locations monitored: {len(aggregated_data)}
    - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Top Affected Locations
    """
    
    for i, (location, data) in enumerate(top_locations, 1):
        count = data.get('total_health_mentions', 0)
        report += f"{i}. {location.title()}: {count} mentions\n    "
    
    report += "\n## Most Common Conditions\n"
    
    for i, (condition, count) in enumerate(top_conditions, 1):
        report += f"{i}. {condition.replace('_', ' ').title()}: {count} mentions\n    "
    
    return report
