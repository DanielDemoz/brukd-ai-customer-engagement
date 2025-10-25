"""
AI-Powered Engagement Recommendations Engine
Brukd AI-Driven Customer Segmentation & Predictive Engagement

This script generates personalized, actionable engagement recommendations
for each customer segment based on their characteristics and risk profiles.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Segment profiles and strategies
SEGMENT_PROFILES = {
    0: {
        'name': 'High-Value Frequent Buyers',
        'characteristics': 'Young professionals, high engagement, frequent purchases',
        'priority': 'Retention & Upselling',
        'churn_risk': 'Low'
    },
    1: {
        'name': 'Budget-Minded Occasional Shoppers',
        'characteristics': 'Price-sensitive, moderate frequency, value-focused',
        'priority': 'Engagement & Conversion',
        'churn_risk': 'Medium'
    },
    2: {
        'name': 'At-Risk Churners',
        'characteristics': 'Low engagement, infrequent purchases, high churn probability',
        'priority': 'Win-Back & Retention',
        'churn_risk': 'High'
    }
}

def load_data():
    """Load data with segments and churn predictions"""
    print("=" * 80)
    print("Loading customer data with predictions...")
    
    df = pd.read_csv('data/data_with_churn_predictions.csv')
    
    print(f"✓ Loaded {len(df):,} customers")
    print(f"\nSegment Distribution:")
    for seg_id, count in df['Cluster'].value_counts().sort_index().items():
        seg_name = SEGMENT_PROFILES[seg_id]['name']
        print(f"  Segment {seg_id} ({seg_name}): {count:,} customers")
    
    return df

def generate_segment_strategies():
    """Generate engagement strategies for each segment"""
    print("\n" + "=" * 80)
    print("ENGAGEMENT STRATEGIES BY SEGMENT")
    print("=" * 80)
    
    strategies = {
        0: {  # High-Value Frequent Buyers
            'primary_actions': [
                "Enroll in VIP/Premium loyalty tier with exclusive benefits",
                "Offer personalized product recommendations based on purchase history",
                "Provide early access to new collections and exclusive sales",
                "Implement referral program with premium incentives",
                "Deploy predictive upselling campaigns for complementary products"
            ],
            'communication_channels': ['Email (personalized)', 'App push', 'SMS', 'Direct mail'],
            'offer_types': ['Exclusive access', 'Premium rewards', 'Personalization', 'VIP events'],
            'frequency': 'Weekly',
            'estimated_impact': '+15-20% revenue per customer',
            'budget_allocation': '30%'
        },
        1: {  # Budget-Minded Occasional Shoppers
            'primary_actions': [
                "Send targeted promotional emails (15-20% discount offers)",
                "Create bundle offers and value packs",
                "Launch seasonal campaigns aligned with purchase patterns",
                "Set up cart abandonment recovery workflows",
                "Introduce loyalty points program for incremental purchases"
            ],
            'communication_channels': ['Email (promotional)', 'Social media', 'Display ads', 'SMS'],
            'offer_types': ['Discounts', 'Bundles', 'Seasonal sales', 'Loyalty points'],
            'frequency': 'Bi-weekly',
            'estimated_impact': '+10-12% conversion rate',
            'budget_allocation': '40%'
        },
        2: {  # At-Risk Churners
            'primary_actions': [
                "🚨 PRIORITY: Immediate win-back campaign activation",
                "Personal outreach from customer success team",
                "Deploy highly personalized incentives (30%+ off first purchase back)",
                "Conduct exit surveys to identify pain points",
                "Implement re-engagement drip campaigns over 30 days",
                "Send \"We miss you\" targeted messaging with compelling offers"
            ],
            'communication_channels': ['Email (urgent)', 'SMS', 'Phone call', 'Retargeting ads'],
            'offer_types': ['Deep discounts (30%+)', 'Free shipping', 'Gift with purchase', 'Surveys'],
            'frequency': 'Immediate, then weekly',
            'estimated_impact': '+12% re-engagement rate',
            'budget_allocation': '30%'
        }
    }
    
    for seg_id, strategy in strategies.items():
        seg_name = SEGMENT_PROFILES[seg_id]['name']
        print(f"\n{'─'*80}")
        print(f"SEGMENT {seg_id}: {seg_name.upper()}")
        print(f"{'─'*80}")
        print(f"Priority: {SEGMENT_PROFILES[seg_id]['priority']}")
        print(f"Churn Risk: {SEGMENT_PROFILES[seg_id]['churn_risk']}")
        print(f"\nPrimary Actions:")
        for i, action in enumerate(strategy['primary_actions'], 1):
            print(f"  {i}. {action}")
        print(f"\nCommunication Channels: {', '.join(strategy['communication_channels'])}")
        print(f"Recommended Frequency: {strategy['frequency']}")
        print(f"Estimated Impact: {strategy['estimated_impact']}")
        print(f"Marketing Budget Allocation: {strategy['budget_allocation']}")
    
    return strategies

def generate_individual_recommendations(df, strategies):
    """Generate personalized recommendations for each customer"""
    print("\n" + "=" * 80)
    print("Generating individual customer recommendations...")
    print("=" * 80)
    
    recommendations = []
    
    for idx, row in df.iterrows():
        segment = int(row['Cluster'])
        segment_name = SEGMENT_PROFILES[segment]['name']
        churn_prob = row['Churn_Probability_90d']
        churn_category = row['Churn_Risk_Category']
        clv = row['CLV_Target']
        
        # Base recommendations from segment strategy
        base_actions = strategies[segment]['primary_actions']
        
        # Customize based on churn risk
        if churn_category == 'High':
            priority = 'URGENT'
            recommended_actions = [
                "🚨 IMMEDIATE WIN-BACK: Personal outreach within 48 hours",
                f"Offer exclusive 35% discount valid for 7 days",
                "Schedule customer success call to understand concerns",
                "Send personalized product recommendations based on past purchases"
            ]
            timeline = 'Immediate action required'
            
        elif churn_category == 'Medium':
            priority = 'HIGH'
            recommended_actions = [
                "Deploy re-engagement email campaign",
                f"Offer 20% discount on next purchase",
                "Highlight new arrivals matching their preferences",
                "Remind of loyalty points/benefits available"
            ]
            timeline = 'Action within 1 week'
            
        else:  # Low risk
            priority = 'STANDARD'
            recommended_actions = base_actions[:3]
            timeline = 'Ongoing engagement'
        
        # Add CLV-based prioritization
        if clv > df['CLV_Target'].quantile(0.75):
            priority_boost = " (HIGH VALUE)"
        else:
            priority_boost = ""
        
        recommendations.append({
            'Customer_ID': row['Customer ID'],
            'Segment': segment,
            'Segment_Name': segment_name,
            'CLV': clv,
            'Churn_Probability_90d': churn_prob,
            'Churn_Risk_Category': churn_category,
            'Priority': priority + priority_boost,
            'Recommended_Actions': ' | '.join(recommended_actions),
            'Communication_Channels': ', '.join(strategies[segment]['communication_channels']),
            'Offer_Type': ', '.join(strategies[segment]['offer_types']),
            'Timeline': timeline,
            'Estimated_Impact': strategies[segment]['estimated_impact']
        })
    
    recommendations_df = pd.DataFrame(recommendations)
    print(f"✓ Generated {len(recommendations_df):,} personalized recommendations")
    
    return recommendations_df

def create_campaign_priorities(recommendations_df):
    """Create prioritized campaign list"""
    print("\nCreating campaign priorities...")
    
    # Priority scoring
    priority_map = {
        'URGENT (HIGH VALUE)': 5,
        'URGENT': 4,
        'HIGH (HIGH VALUE)': 3,
        'HIGH': 2,
        'STANDARD (HIGH VALUE)': 1,
        'STANDARD': 0
    }
    
    recommendations_df['Priority_Score'] = recommendations_df['Priority'].map(priority_map)
    recommendations_df = recommendations_df.sort_values(['Priority_Score', 'CLV'], ascending=[False, False])
    
    # Campaign summary
    print("\n" + "=" * 80)
    print("CAMPAIGN PRIORITIES SUMMARY")
    print("=" * 80)
    
    for priority in sorted(recommendations_df['Priority'].unique(), 
                          key=lambda x: priority_map.get(x, -1), reverse=True):
        count = (recommendations_df['Priority'] == priority).sum()
        total_clv = recommendations_df[recommendations_df['Priority'] == priority]['CLV'].sum()
        print(f"\n{priority}:")
        print(f"  Customers: {count:,}")
        print(f"  Total CLV at Risk: ${total_clv:,.2f}")
        print(f"  Recommended Action: {recommendations_df[recommendations_df['Priority'] == priority]['Timeline'].iloc[0]}")
    
    return recommendations_df

def generate_campaign_templates(strategies):
    """Generate email/message templates for each segment"""
    print("\nGenerating campaign templates...")
    
    templates = {
        0: {  # High-Value Frequent Buyers
            'email_subject': '🌟 Exclusive VIP Access: You\'re Invited!',
            'email_body': '''
Dear Valued Customer,

Thank you for being one of our most loyal customers! We've noticed your exceptional engagement with our brand, and we'd like to show our appreciation.

🎁 EXCLUSIVE FOR YOU:
• Early access to our Spring Collection (48 hours before public launch)
• Complimentary premium gift with your next purchase
• Personal styling session with our expert team
• 20% off your next purchase + free express shipping

Your loyalty means everything to us. As a VIP member, you'll continue to receive exclusive perks and first access to our best offers.

Shop your exclusive preview now: [LINK]

Best regards,
The Brukd Team
            ''',
            'sms': 'VIP Alert! 🌟 Early access to new collection + 20% off just for you. Shop now: [LINK]'
        },
        1: {  # Budget-Minded Occasional Shoppers
            'email_subject': '💰 Special Offer: 20% Off Everything You Love!',
            'email_body': '''
Hi there!

We've picked out some amazing deals just for you based on what you love.

🎉 THIS WEEKEND ONLY:
• 20% OFF your favorite categories
• FREE shipping on orders over $50
• Extra 10% off when you buy 2+ items

Don't miss out on these incredible savings!

[SHOP NOW]

P.S. Join our loyalty program and earn points on every purchase!

Happy Shopping!
            ''',
            'sms': '💰 Weekend Sale! 20% off + free shipping $50+. Limited time: [LINK]'
        },
        2: {  # At-Risk Churners
            'email_subject': '❤️ We Miss You! Here\'s 35% Off to Welcome You Back',
            'email_body': '''
We Miss You!

It's been a while since we've seen you, and we'd love to welcome you back.

We've made some exciting changes and added new products we think you'll love.

🎁 SPECIAL WELCOME BACK OFFER:
• 35% OFF your entire order
• FREE shipping (no minimum)
• Bonus gift with purchase
• Valid for 7 days only

We value your feedback! Take our 2-minute survey and get an extra $10 credit: [SURVEY LINK]

Come back and see what's new: [SHOP NOW]

We'd love to have you back!
            ''',
            'sms': '❤️ We miss you! Here\'s 35% OFF to welcome you back. 7 days only: [LINK]'
        }
    }
    
    # Save templates
    templates_df = pd.DataFrame([
        {
            'Segment': seg_id,
            'Segment_Name': SEGMENT_PROFILES[seg_id]['name'],
            'Email_Subject': template['email_subject'],
            'Email_Body': template['email_body'].strip(),
            'SMS_Template': template['sms']
        }
        for seg_id, template in templates.items()
    ])
    
    templates_df.to_csv('data/campaign_templates.csv', index=False)
    print("✓ Saved: data/campaign_templates.csv")
    
    return templates

def save_recommendations(recommendations_df):
    """Save all recommendations"""
    print("\nSaving engagement recommendations...")
    
    # Save full recommendations
    recommendations_df.to_csv('data/customer_engagement_recommendations.csv', index=False)
    
    # Save high-priority customers for immediate action
    high_priority = recommendations_df[
        recommendations_df['Priority'].str.contains('URGENT|HIGH')
    ].copy()
    high_priority.to_csv('data/high_priority_customers.csv', index=False)
    
    print(f"✓ Saved: data/customer_engagement_recommendations.csv ({len(recommendations_df):,} customers)")
    print(f"✓ Saved: data/high_priority_customers.csv ({len(high_priority):,} priority customers)")
    
    return high_priority

def generate_executive_summary(df, recommendations_df):
    """Generate executive summary of recommendations"""
    print("\n" + "=" * 80)
    print(" " * 25 + "EXECUTIVE SUMMARY")
    print("=" * 80)
    
    total_customers = len(df)
    total_clv = df['CLV_Target'].sum()
    
    print(f"\n📊 CUSTOMER BASE OVERVIEW:")
    print(f"  Total Customers: {total_customers:,}")
    print(f"  Total Customer Lifetime Value: ${total_clv:,.2f}")
    print(f"  Average CLV: ${total_clv/total_customers:,.2f}")
    
    print(f"\n🎯 ENGAGEMENT PRIORITIES:")
    urgent_customers = recommendations_df[recommendations_df['Priority'].str.contains('URGENT')].shape[0]
    high_priority_customers = recommendations_df[recommendations_df['Priority'].str.contains('HIGH')].shape[0]
    urgent_clv = recommendations_df[recommendations_df['Priority'].str.contains('URGENT')]['CLV'].sum()
    
    print(f"  Urgent Action Required: {urgent_customers:,} customers (${urgent_clv:,.2f} at risk)")
    print(f"  High Priority: {high_priority_customers:,} customers")
    print(f"  Standard Engagement: {total_customers - urgent_customers - high_priority_customers:,} customers")
    
    print(f"\n📈 EXPECTED IMPACT:")
    print(f"  High-Value Segment: +15-20% revenue increase")
    print(f"  Budget-Minded Segment: +10-12% conversion improvement")
    print(f"  At-Risk Segment: +12% re-engagement rate")
    print(f"  Overall Projected Re-engagement Lift: +12%")
    
    print(f"\n💰 MARKETING BUDGET ALLOCATION:")
    print(f"  High-Value Frequent Buyers: 30% of budget")
    print(f"  Budget-Minded Shoppers: 40% of budget")
    print(f"  At-Risk Churners: 30% of budget")
    
    print(f"\n✅ IMMEDIATE ACTIONS:")
    print(f"  1. Launch urgent win-back campaign for {urgent_customers:,} at-risk customers")
    print(f"  2. Activate VIP program for high-value segment")
    print(f"  3. Deploy promotional campaign to budget-minded segment")
    print(f"  4. Set up automated engagement workflows by segment")
    
    print("\n" + "=" * 80)
    print("✓ Engagement recommendations ready for deployment!")
    print("=" * 80)

def main():
    """Main execution function"""
    print("=" * 80)
    print(" " * 15 + "BRUKD AI-DRIVEN CUSTOMER ENGAGEMENT")
    print(" " * 20 + "Engagement Recommendations Engine")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Generate segment strategies
    strategies = generate_segment_strategies()
    
    # Generate individual recommendations
    recommendations_df = generate_individual_recommendations(df, strategies)
    
    # Create campaign priorities
    recommendations_df = create_campaign_priorities(recommendations_df)
    
    # Generate campaign templates
    templates = generate_campaign_templates(strategies)
    
    # Save recommendations
    high_priority = save_recommendations(recommendations_df)
    
    # Executive summary
    generate_executive_summary(df, recommendations_df)
    
    print("\n✓ Engagement recommendations engine complete!")
    
    return recommendations_df, templates

if __name__ == "__main__":
    recommendations_df, templates = main()

