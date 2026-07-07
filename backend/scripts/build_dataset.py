#!/usr/bin/env python3
"""
Build (or regenerate) the email/reply dataset.

The seed data below IS the source of truth — it's checked into the repo so the
system runs without any API key for this step.

Usage:
    python backend/scripts/build_dataset.py                # write seed data
    python backend/scripts/build_dataset.py --augment 5     # + 5 Groq-generated pairs

Design note (dataset §4): these are hand-authored to guarantee clean single-
email→single-reply pairing with deliberate category/tone coverage.  They are
NOT scraped from Enron or any public corpus (PII/licensing concerns, messy
threading).  See README for the representativeness argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Seed data: ~36 pairs across 6 categories, 6 each ─────────────────────────
SEED_DATA: list[dict] = [
    # ── customer_support (6) ──────────────────────────────────────────────
    {
        "id": "cs_001",
        "category": "customer_support",
        "incoming_email": (
            "Subject: Order #8812 not received\n\n"
            "Hi,\n\nI placed order #8812 on June 15th and the tracking still shows "
            "'label created.' It's been two weeks now and I really need this for a "
            "project deadline on July 5th. Can you look into this?\n\nThanks,\nMaria Chen"
        ),
        "sent_reply": (
            "Hi Maria,\n\nThank you for reaching out about order #8812. I'm sorry for "
            "the delay — that's not the experience we want you to have.\n\nI've escalated "
            "this with our shipping partner and requested an urgent trace on your package. "
            "You'll receive an updated tracking link within 24 hours. If the package can't "
            "be located, we'll expedite a replacement shipment via overnight delivery to "
            "ensure it arrives before your July 5th deadline.\n\nPlease don't hesitate to "
            "reply if you have any other questions.\n\nBest regards,\nCustomer Support Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "cs_002",
        "category": "customer_support",
        "incoming_email": (
            "Subject: How do I reset my password?\n\n"
            "Hey, I forgot my password and the reset link isn't arriving in my inbox. "
            "I've checked spam too. My account email is jpark@email.com. Help?\n\n- Jamie"
        ),
        "sent_reply": (
            "Hi Jamie,\n\nThanks for letting us know. I've manually triggered a password "
            "reset for jpark@email.com — please check your inbox (and spam folder) in the "
            "next few minutes.\n\nIf it still doesn't arrive, it's possible the account is "
            "registered under a different email address. In that case, reply with your "
            "username and I can look up the associated email.\n\nCheers,\nSupport Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "medium"},
    },
    {
        "id": "cs_003",
        "category": "customer_support",
        "incoming_email": (
            "Subject: Defective product — want a refund\n\n"
            "To whom it may concern,\n\nThe BlendMax Pro I received (order #7234) arrived "
            "with a cracked lid and the motor makes a grinding noise. I'd like a full "
            "refund processed to my original payment method. I've attached photos of the "
            "damage.\n\nRegards,\nDavid Okafor"
        ),
        "sent_reply": (
            "Dear David,\n\nI'm very sorry to hear about the condition of your BlendMax "
            "Pro. That's clearly not acceptable, and I apologize for the inconvenience.\n\n"
            "I've initiated a full refund of $89.99 to your original payment method. You "
            "should see it reflected within 5–7 business days. There's no need to return "
            "the defective unit — please dispose of it at your convenience.\n\nIf you'd "
            "like a replacement instead, I'm happy to arrange that at no charge. Just let "
            "me know.\n\nSincerely,\nCustomer Support Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "cs_004",
        "category": "customer_support",
        "incoming_email": (
            "Subject: Can I change my subscription plan?\n\n"
            "Hi there! I'm currently on the Basic plan ($9/month) but want to upgrade "
            "to Pro. Will I be charged the full $29 this month or just the prorated "
            "difference? Also, do I keep my existing data?\n\nThanks!\nLisa"
        ),
        "sent_reply": (
            "Hi Lisa!\n\nGreat question — when you upgrade mid-cycle, you'll only be "
            "charged the prorated difference for the remaining days this month. Starting "
            "next billing cycle, you'll be charged the full $29/month.\n\nAll your existing "
            "data, projects, and settings will carry over seamlessly. You'll immediately "
            "get access to Pro features like advanced analytics and priority support.\n\n"
            "You can upgrade directly from Settings → Subscription, or I can process it "
            "on my end right now if you'd like. Just say the word!\n\nBest,\nSupport Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "low"},
    },
    {
        "id": "cs_005",
        "category": "customer_support",
        "incoming_email": (
            "Subject: App crashes on startup\n\n"
            "Your mobile app (v3.2.1) crashes immediately after the splash screen on my "
            "iPhone 14 running iOS 17.5. I've tried reinstalling twice. This started "
            "after your last update.\n\nPlease fix ASAP — I rely on this for work.\n\nMark T."
        ),
        "sent_reply": (
            "Hi Mark,\n\nThank you for the detailed report — knowing the exact version "
            "(3.2.1) and your device info is really helpful.\n\nWe've identified a crash "
            "affecting a subset of iOS 17.5 devices and a hotfix (v3.2.2) is currently in "
            "App Store review. It should be available within 24–48 hours.\n\nIn the meantime, "
            "you can use our web app at app.example.com as a workaround — it has full "
            "feature parity.\n\nI'll follow up once the fix is live. Sorry for the "
            "disruption!\n\nBest,\nTechnical Support"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "cs_006",
        "category": "customer_support",
        "incoming_email": (
            "Subject: Shipping to PO Box?\n\n"
            "Quick question — do you ship to PO Boxes? I live in a rural area and "
            "that's my only mailing option.\n\nThanks,\nRachel"
        ),
        "sent_reply": (
            "Hi Rachel,\n\nYes, we do ship to PO Boxes! We'll use USPS for your delivery "
            "since they're the only carrier that delivers to PO Boxes. Standard shipping "
            "times (5–7 business days) apply.\n\nJust enter your PO Box address at checkout "
            "and the system will handle the rest. If you run into any issues, let me "
            "know and I'll place the order manually.\n\nHave a great day!\nSupport Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "low"},
    },

    # ── sales_inquiry (6) ─────────────────────────────────────────────────
    {
        "id": "si_001",
        "category": "sales_inquiry",
        "incoming_email": (
            "Subject: Enterprise pricing for 200+ seats\n\n"
            "Hello,\n\nI'm the IT Director at Greenfield Manufacturing. We're evaluating "
            "your platform for company-wide deployment (~200 users). Could you send over "
            "enterprise pricing and information about volume discounts, SSO integration, "
            "and dedicated support options?\n\nBest,\nRobert Huang\nIT Director, Greenfield Mfg."
        ),
        "sent_reply": (
            "Dear Robert,\n\nThank you for considering us for Greenfield Manufacturing — "
            "we'd be thrilled to support a deployment of that scale.\n\nFor 200+ seats, our "
            "Enterprise plan includes:\n• Volume pricing at $18/user/month (vs. standard $29)\n"
            "• SAML-based SSO integration with your existing identity provider\n• Dedicated "
            "Customer Success Manager\n• 99.9% SLA with priority support (4-hour response)\n\n"
            "I'd love to schedule a 30-minute call to discuss your specific requirements. "
            "Would any time this Thursday or Friday work for you?\n\nI'm also attaching our "
            "Enterprise overview deck for your review.\n\nBest regards,\nSarah Mitchell\n"
            "Enterprise Sales"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "medium"},
    },
    {
        "id": "si_002",
        "category": "sales_inquiry",
        "incoming_email": (
            "Subject: Do you offer a free trial?\n\n"
            "Hi! I saw your product demo on YouTube and it looks great. Do you have a "
            "free trial so I can test it with my team of 5 before committing? We're a "
            "small marketing agency.\n\nCheers,\nAlex"
        ),
        "sent_reply": (
            "Hey Alex!\n\nGlad the demo caught your eye! Yes — we offer a 14-day free "
            "trial with full access to all features, no credit card required.\n\nFor a team "
            "of 5, I'd recommend starting with our Team plan trial, which includes "
            "collaboration features and shared workspaces. I've set up a trial workspace "
            "for you — just click the link below to get started:\n\n→ [Trial Activation Link]\n\n"
            "I'll also send a quick-start guide tailored for marketing agencies. If you "
            "have any questions during the trial, you can reach me directly.\n\n"
            "Looking forward to hearing what you think!\nAlex R.\nSales Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "low"},
    },
    {
        "id": "si_003",
        "category": "sales_inquiry",
        "incoming_email": (
            "Subject: Comparison with CompetitorX\n\n"
            "We're currently using CompetitorX but their API rate limits are killing us. "
            "How does your API compare? Specifically: rate limits, webhook support, and "
            "can we do bulk operations (1000+ records at a time)?\n\nThanks,\nPriya Sharma\n"
            "CTO, DataFlow Inc."
        ),
        "sent_reply": (
            "Hi Priya,\n\nGreat questions — API capability is one of our strongest "
            "differentiators. Here's how we compare:\n\n• Rate limits: 10,000 requests/min "
            "on our Growth plan (vs. CompetitorX's 1,000/min)\n• Webhooks: Full support with "
            "configurable retry logic and event filtering\n• Bulk operations: Our batch API "
            "handles up to 5,000 records per call with async processing\n\nWe also offer a "
            "migration tool that can import your existing CompetitorX data in one click.\n\n"
            "Would a technical deep-dive call with our engineering team be useful? I can "
            "set one up this week.\n\nBest,\nSales Engineering Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "medium"},
    },
    {
        "id": "si_004",
        "category": "sales_inquiry",
        "incoming_email": (
            "Subject: Partnership opportunity\n\n"
            "Hi,\n\nI run a SaaS consulting firm and several of my clients could benefit "
            "from your platform. Do you have a partner/reseller program? What are the "
            "commission structures and co-marketing support like?\n\nBest,\nTom Eriksen"
        ),
        "sent_reply": (
            "Hi Tom,\n\nThanks for reaching out — we're always excited to connect with "
            "consultants in the SaaS space.\n\nYes, we have an active Partner Program:\n"
            "• 20% recurring commission on referred accounts\n• Co-branded landing pages "
            "and marketing materials\n• Partner portal with deal registration and pipeline "
            "tracking\n• Quarterly co-marketing budget for joint webinars/content\n\n"
            "I'd love to learn more about your client base to see how we can tailor the "
            "partnership. Are you available for a call next week?\n\nI'll send the full "
            "Partner Program guide in the meantime.\n\nBest regards,\nPartnership Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "vendor", "urgency": "low"},
    },
    {
        "id": "si_005",
        "category": "sales_inquiry",
        "incoming_email": (
            "Subject: Nonprofit discount?\n\n"
            "Hello, I work for a nonprofit education foundation. We're interested in your "
            "platform for our 50 staff members but our budget is very limited. Do you "
            "offer nonprofit pricing?\n\nThank you,\nCarla Reyes\nOperations Manager"
        ),
        "sent_reply": (
            "Dear Carla,\n\nThank you for the important work your foundation does in "
            "education. We're happy to support nonprofit organizations.\n\nWe offer a 50% "
            "discount on all plans for registered nonprofits. For 50 users, that would be "
            "$7.50/user/month on our Team plan (normally $15/user/month).\n\nTo apply, just "
            "send us a copy of your 501(c)(3) determination letter or equivalent, and I'll "
            "activate the discount on your account immediately.\n\nPlease let me know if "
            "you have any other questions.\n\nWarm regards,\nSales Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "low"},
    },
    {
        "id": "si_006",
        "category": "sales_inquiry",
        "incoming_email": (
            "Subject: Re: Demo follow-up\n\n"
            "Thanks for the demo yesterday! The team liked what they saw. Two quick follow-ups:\n"
            "1. Can we get a sandbox environment to test our API integration?\n"
            "2. What's your data residency situation? We need EU hosting (GDPR).\n\nBest,\nKai"
        ),
        "sent_reply": (
            "Hi Kai,\n\nReally glad the team enjoyed the demo! To your questions:\n\n"
            "1. Sandbox: Absolutely — I've provisioned a full-featured sandbox environment "
            "for your team. Credentials are in the separate email I just sent. It includes "
            "sample data and our API playground.\n\n2. Data residency: We offer EU-hosted "
            "instances (Frankfurt data center) on our Growth and Enterprise plans. All data "
            "stays within the EU, and we're fully GDPR-compliant with DPA available on "
            "request.\n\nLet me know how the sandbox testing goes — happy to jump on a call "
            "if your devs have technical questions.\n\nBest,\nSales Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "medium"},
    },

    # ── scheduling (6) ────────────────────────────────────────────────────
    {
        "id": "sc_001",
        "category": "scheduling",
        "incoming_email": (
            "Subject: Q3 planning meeting\n\n"
            "Hi team,\n\nWe need to schedule our Q3 planning session. It should be 2 hours "
            "and include all department heads. Can we target the week of July 14th? "
            "Please send your availability.\n\nThanks,\nDirector of Operations"
        ),
        "sent_reply": (
            "Hi,\n\nThanks for kicking this off. I'm available the week of July 14th on:\n"
            "• Tuesday 7/15: 10am–12pm or 2pm–4pm\n• Wednesday 7/16: 9am–11am\n"
            "• Thursday 7/17: 1pm–3pm\n\nI'd suggest the Tuesday morning slot since "
            "it gives us the full week afterward to follow up on action items before "
            "the weekend.\n\nI'll reserve Conference Room A (capacity 12) provisionally "
            "for Tuesday 10am — let me know if another time works better for the group.\n\n"
            "Best,\nOps Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "colleague", "urgency": "medium"},
    },
    {
        "id": "sc_002",
        "category": "scheduling",
        "incoming_email": (
            "Subject: Reschedule Thursday's 1:1?\n\n"
            "Hey, something came up Thursday afternoon — can we move our 1:1 to Friday "
            "morning instead? Anytime before noon works.\n\n- Sam"
        ),
        "sent_reply": (
            "Hey Sam,\n\nNo problem at all! Let's do Friday at 10am — I'll update the "
            "calendar invite. Same agenda applies.\n\nSee you then!\n- Ops"
        ),
        "metadata": {"tone": "casual", "sender_role": "colleague", "urgency": "low"},
    },
    {
        "id": "sc_003",
        "category": "scheduling",
        "incoming_email": (
            "Subject: Client onboarding kickoff — scheduling\n\n"
            "Hi,\n\nWe just signed Meridian Corp and need to schedule the onboarding "
            "kickoff within the next 5 business days. Attendees: our PM, tech lead, "
            "and their VP of Product (Sandra Lee). The client is in PST.\n\n"
            "Can you coordinate?\n\nThanks,\nAccount Manager"
        ),
        "sent_reply": (
            "Hi,\n\nCongrats on closing Meridian! I'll coordinate the kickoff. Here's my "
            "plan:\n\n1. I'll reach out to Sandra Lee's office today to get her availability\n"
            "2. I've blocked tentative 1-hour slots on our PM's and tech lead's calendars "
            "for Mon–Wed next week (all in the 10am–2pm PST window)\n3. Once Sandra confirms, "
            "I'll send the final invite with a Zoom link and agenda\n\nI'll aim to have this "
            "locked down by end of day tomorrow. Will keep you posted.\n\nBest,\nOps Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "colleague", "urgency": "high"},
    },
    {
        "id": "sc_004",
        "category": "scheduling",
        "incoming_email": (
            "Subject: Team lunch next week\n\n"
            "Hey everyone! 🎉\n\nTo celebrate shipping v2.0, let's do a team lunch next "
            "week. Any day works for me. Italian or sushi? Drop your votes and availability!\n\n"
            "- Jordan"
        ),
        "sent_reply": (
            "Hey Jordan!\n\nLove this idea — congrats everyone! 🎉\n\nMy vote: sushi! I'm "
            "free any day except Wednesday. How about Thursday around 12:30pm? There's a "
            "great new sushi place on Market Street that does group reservations.\n\n"
            "Happy to make the reservation once we have a headcount. Everyone please "
            "reply-all with your preference and availability!\n\n- Ops"
        ),
        "metadata": {"tone": "casual", "sender_role": "colleague", "urgency": "low"},
    },
    {
        "id": "sc_005",
        "category": "scheduling",
        "incoming_email": (
            "Subject: Board meeting prep — time-sensitive\n\n"
            "The board meeting is August 1st. I need the following by July 25th:\n"
            "• Financial summary from Finance\n• Product roadmap update from Product\n"
            "• Hiring plan from HR\n\nPlease schedule individual prep sessions with each "
            "department head this week.\n\nThanks,\nCEO"
        ),
        "sent_reply": (
            "Hi,\n\nUnderstood — I'll have all prep sessions scheduled today. Here's the "
            "plan:\n\n• Finance (CFO): Scheduling for Tuesday or Wednesday this week\n"
            "• Product (VP Product): Scheduling for Wednesday or Thursday\n"
            "• HR (HR Director): Scheduling for Thursday or Friday\n\n"
            "Each session will be 45 minutes. I'll send calendar invites by end of today "
            "with a brief agenda and the July 25th deliverable deadline noted.\n\n"
            "I'll also create a shared folder for all board prep materials so everything "
            "is in one place.\n\nBest regards,\nExecutive Assistant"
        ),
        "metadata": {"tone": "formal", "sender_role": "colleague", "urgency": "high"},
    },
    {
        "id": "sc_006",
        "category": "scheduling",
        "incoming_email": (
            "Subject: Vacation coverage\n\n"
            "Hi, I'll be on vacation July 21–28. Can someone cover my Monday and Thursday "
            "standups? They're at 9:30am, 15 minutes each. I'll leave notes on what to "
            "report.\n\nThanks!\n- Nina"
        ),
        "sent_reply": (
            "Hi Nina,\n\nEnjoy your vacation! I'll cover both standups (Mon 7/21 and "
            "Thu 7/24, 9:30am). Just share your notes by Friday and I'll make sure "
            "everything's reported accurately.\n\nI'll also keep an eye on your Slack "
            "channels for anything urgent. If there's a specific escalation contact, "
            "let me know.\n\nHave a great time!\n- Ops"
        ),
        "metadata": {"tone": "casual", "sender_role": "colleague", "urgency": "medium"},
    },

    # ── billing_invoice (6) ───────────────────────────────────────────────
    {
        "id": "bi_001",
        "category": "billing_invoice",
        "incoming_email": (
            "Subject: Invoice #INV-2024-0892 discrepancy\n\n"
            "Hello,\n\nI'm reviewing invoice #INV-2024-0892 dated June 30th and there's a "
            "charge for 'Premium Add-on' at $49/month that we never authorized. Our "
            "contract only covers the Standard plan. Please correct this and issue a "
            "revised invoice.\n\nRegards,\nAmir Patel\nFinance Manager, TechCorp"
        ),
        "sent_reply": (
            "Dear Amir,\n\nThank you for catching that. I've reviewed your account and "
            "you're correct — the Premium Add-on charge on invoice #INV-2024-0892 was "
            "applied in error.\n\nI've taken the following steps:\n1. Removed the $49 "
            "Premium Add-on charge\n2. Issued a corrected invoice (#INV-2024-0892-R1), "
            "attached to this email\n3. Applied a $49 credit to your next billing cycle "
            "as a goodwill gesture\n\nThe revised total is $299.00 (Standard plan only). "
            "Please let me know if everything looks correct.\n\nApologies for the "
            "inconvenience.\n\nBest regards,\nBilling Department"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "bi_002",
        "category": "billing_invoice",
        "incoming_email": (
            "Subject: Need W-9 for our records\n\n"
            "Hi,\n\nOur accounting team needs an updated W-9 form from your company before "
            "we can process payment on PO #4401. Can you send one over at your earliest "
            "convenience?\n\nThanks,\nAccounts Payable"
        ),
        "sent_reply": (
            "Hi,\n\nOf course — I've attached our current W-9 form to this email. It was "
            "last updated in January 2024 and all information is current.\n\nFor your "
            "records:\n• Legal entity: Acme Solutions LLC\n• EIN: XX-XXXXXXX (on the W-9)\n"
            "• Address matches what's on PO #4401\n\nPlease let me know once payment on "
            "PO #4401 is processed, or if you need any additional documentation.\n\n"
            "Best,\nFinance Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "medium"},
    },
    {
        "id": "bi_003",
        "category": "billing_invoice",
        "incoming_email": (
            "Subject: Payment failed — card expired\n\n"
            "Hey, I just got an email saying my payment failed because my card expired. "
            "I've updated my card info in the dashboard. Can you retry the charge? Don't "
            "want my account to get suspended.\n\n- Tony"
        ),
        "sent_reply": (
            "Hey Tony,\n\nThanks for updating your card info so quickly! I've manually "
            "retried the charge and it went through successfully. Your account is in "
            "good standing — no interruption to your service.\n\nHere's your updated "
            "receipt: [Receipt Link]\n\nTip: you can set up a backup payment method in "
            "Settings → Billing to avoid this in the future.\n\nCheers,\nBilling Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "bi_004",
        "category": "billing_invoice",
        "incoming_email": (
            "Subject: Annual billing switch\n\n"
            "Hi, I'm currently paying monthly ($29/month = $348/year). I see annual "
            "billing is $249/year — that's a significant saving. Can I switch mid-cycle? "
            "How does the transition work?\n\nThanks,\nElena"
        ),
        "sent_reply": (
            "Hi Elena,\n\nGreat eye — annual billing saves you about 28%! Here's how "
            "the switch works:\n\n1. We'll prorate a credit for the unused portion of your "
            "current monthly cycle\n2. Apply that credit toward the $249 annual charge\n"
            "3. Your new billing anniversary becomes today's date\n\nSo you'd pay $249 "
            "minus your prorated credit today, and your next charge would be in 12 months.\n\n"
            "Want me to process the switch now? Just confirm and I'll take care of it.\n\n"
            "Best,\nBilling Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "low"},
    },
    {
        "id": "bi_005",
        "category": "billing_invoice",
        "incoming_email": (
            "Subject: Tax exemption certificate\n\n"
            "Hello,\n\nWe are a tax-exempt government entity (certificate attached). "
            "Please update our account to reflect our tax-exempt status so future "
            "invoices do not include sales tax.\n\nRegards,\nState Dept. of Education"
        ),
        "sent_reply": (
            "Dear State Department of Education,\n\nThank you for providing your tax "
            "exemption certificate. I've verified the documentation and updated your "
            "account accordingly.\n\nEffective immediately:\n• All future invoices will "
            "exclude sales tax\n• I've also issued a credit for tax charged on your most "
            "recent invoice (#INV-2024-0845) — the credit memo is attached\n\nThe exemption "
            "certificate will remain on file. Please notify us if it's renewed or updated.\n\n"
            "Best regards,\nBilling Department"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "medium"},
    },
    {
        "id": "bi_006",
        "category": "billing_invoice",
        "incoming_email": (
            "Subject: Duplicate charge on my account\n\n"
            "I was charged twice on June 28 — $29 each time (transaction IDs: TXN-8891 "
            "and TXN-8892). Please refund the duplicate. My account ID is ACC-55123.\n\n"
            "Thanks,\nMike"
        ),
        "sent_reply": (
            "Hi Mike,\n\nYou're right — I can see both charges (TXN-8891 and TXN-8892) "
            "on account ACC-55123. The duplicate was caused by a processing hiccup on "
            "our end.\n\nI've refunded $29 for TXN-8892 (the duplicate). It should appear "
            "on your statement within 3–5 business days.\n\nSorry about that! Your account "
            "is all set now. Let me know if you see any other issues.\n\nBest,\nBilling Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "high"},
    },

    # ── complaint_escalation (6) ──────────────────────────────────────────
    {
        "id": "ce_001",
        "category": "complaint_escalation",
        "incoming_email": (
            "Subject: UNACCEPTABLE service — escalate immediately\n\n"
            "I have been trying to resolve a billing issue for THREE WEEKS. I've spoken "
            "to four different agents, been given conflicting information each time, and "
            "still haven't received my $200 refund. This is absolutely unacceptable. "
            "I want to speak with a manager TODAY.\n\nFurious,\nKaren Mitchell"
        ),
        "sent_reply": (
            "Dear Karen,\n\nI sincerely apologize for this frustrating experience. Three "
            "weeks without resolution — and conflicting information on top of that — is "
            "completely unacceptable, and I understand your frustration.\n\nI've escalated "
            "your case directly to our Customer Experience Manager, James Torres. Here's "
            "what's happening right now:\n\n1. James will call you personally within 2 hours "
            "at the number on file\n2. I've processed your $200 refund immediately (confirmation "
            "#RF-90234) — expect it in 3–5 business days\n3. I've applied a $50 account credit "
            "for the inconvenience\n\nIf you'd prefer a specific callback time or phone number, "
            "please reply and we'll accommodate.\n\nAgain, I'm deeply sorry for this "
            "experience.\n\nSincerely,\nSenior Support Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "ce_002",
        "category": "complaint_escalation",
        "incoming_email": (
            "Subject: Data loss after your update\n\n"
            "Your latest platform update deleted 3 months of our project data. We have "
            "a team of 20 relying on this. This is a critical business impact. We need "
            "the data restored NOW and an explanation of what happened.\n\nUrgently,\n"
            "Daniel Foster\nVP Engineering, BuildRight Inc."
        ),
        "sent_reply": (
            "Dear Daniel,\n\nThis is a critical issue and it has our full attention. I want "
            "to assure you that data restoration is already in progress.\n\nHere's our "
            "immediate action plan:\n\n1. Our engineering team has identified the issue — a "
            "migration script in the June 28 update incorrectly archived active projects\n"
            "2. Your data is NOT permanently deleted — it's in our backup archive\n"
            "3. We've begun restoring all affected projects and estimate full restoration "
            "within 4 hours\n4. Our CTO, Amanda Chen, will email you a formal incident "
            "report within 24 hours\n\nI'll send you progress updates every hour until "
            "restoration is complete. If you need to reach me directly: [direct line].\n\n"
            "We take full responsibility for this disruption.\n\nSincerely,\n"
            "Director of Engineering Support"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "ce_003",
        "category": "complaint_escalation",
        "incoming_email": (
            "Subject: Rude support agent — complaint\n\n"
            "I just got off a chat with support agent 'Ryan' who was condescending and "
            "dismissive about my issue. He basically told me 'that's just how it works' "
            "when the feature clearly isn't working as documented. I want this reported "
            "and I want someone competent to help me.\n\n- Angela"
        ),
        "sent_reply": (
            "Hi Angela,\n\nI'm really sorry about that interaction — that's not at all how "
            "we want our customers treated, and it's not the standard we hold ourselves to.\n\n"
            "I've done two things:\n1. Reported this to our Support Manager, who will review "
            "the chat transcript and address it with the agent directly\n2. Assigned your "
            "original technical issue to Sarah, one of our senior support specialists — "
            "she'll be reaching out within the hour to resolve it properly\n\n"
            "Could you share your original ticket number so Sarah has all the context? "
            "We'll make sure this gets resolved correctly this time.\n\nThank you for "
            "letting us know.\n\nBest,\nCustomer Experience Team"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "ce_004",
        "category": "complaint_escalation",
        "incoming_email": (
            "Subject: Service outage cost us a major deal\n\n"
            "Your platform went down for 6 hours during our live product demo to a Fortune "
            "500 prospect. We lost the deal. We're on your $500/month Enterprise plan with "
            "a 99.9% SLA. I expect compensation and an explanation.\n\nRegards,\n"
            "Patricia Coleman\nCEO, SalesForward"
        ),
        "sent_reply": (
            "Dear Patricia,\n\nI understand the severity of this situation and I'm deeply "
            "sorry for the impact on your business. Losing a deal due to our outage is "
            "exactly the scenario our SLA exists to prevent.\n\nImmediate actions:\n\n"
            "1. SLA credit: Per our 99.9% uptime guarantee, the 6-hour outage entitles "
            "you to a full month's credit ($500). I've applied this to your account "
            "effective immediately\n2. Additional compensation: Given the business impact, "
            "I'd like to discuss further accommodation — I've asked our VP of Customer "
            "Success, Martin Hall, to schedule a call with you this week\n3. Incident report: "
            "A detailed post-mortem is attached, including root cause (database failover "
            "failure) and the preventive measures we've implemented\n\nThis is not the "
            "service you're paying for, and we take full responsibility.\n\n"
            "Sincerely,\nHead of Customer Success"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "ce_005",
        "category": "complaint_escalation",
        "incoming_email": (
            "Subject: Misleading pricing page\n\n"
            "Your pricing page says 'unlimited storage' on the Pro plan but the fine print "
            "in the ToS says it's capped at 100GB. That's misleading. I upgraded "
            "specifically for unlimited storage and now I'm hitting the cap. Fix this or "
            "refund my last 3 months.\n\n- Derek"
        ),
        "sent_reply": (
            "Hi Derek,\n\nYou raise a completely valid point — 'unlimited' and '100GB cap' "
            "are contradictory, and I understand the frustration of hitting a limit you "
            "didn't expect.\n\nHere's what I've done:\n\n1. Increased your storage limit to "
            "500GB immediately (true unlimited is being rolled out next quarter)\n2. Flagged "
            "the pricing page discrepancy with our marketing team for correction\n3. Refunded "
            "your last 3 months ($87) as requested — confirmation #RF-10492\n\nYou shouldn't "
            "have to read fine print to get what was advertised. We'll update the pricing page "
            "language within this week.\n\nI appreciate you holding us accountable.\n\n"
            "Best,\nCustomer Experience Team"
        ),
        "metadata": {"tone": "casual", "sender_role": "customer", "urgency": "high"},
    },
    {
        "id": "ce_006",
        "category": "complaint_escalation",
        "incoming_email": (
            "Subject: Fourth time contacting about the same issue\n\n"
            "This is the FOURTH email I've sent about my account being locked out. Every "
            "time someone says 'it's fixed' and every time it happens again within days. "
            "Ticket numbers: #3321, #3345, #3367. Either fix it permanently or I'm canceling "
            "and filing a complaint with the BBB.\n\nDone with this,\nSteve"
        ),
        "sent_reply": (
            "Hi Steve,\n\nI completely understand your frustration — four contacts for the "
            "same unresolved issue is a failure on our end, plain and simple.\n\nI've reviewed "
            "all three tickets (#3321, #3345, #3367) and found the root cause: your account "
            "has a conflicting legacy security rule that keeps re-triggering the lockout after "
            "each manual unlock.\n\nHere's what's different this time:\n1. I've removed the "
            "legacy rule entirely (not just unlocked your account)\n2. Verified your login "
            "works with our QA team\n3. Set up a monitoring alert — if your account locks "
            "again in the next 30 days, I'll be notified personally and will fix it within "
            "1 hour\n\nI'm also applying 2 months of free service for the repeated hassle.\n\n"
            "You shouldn't have had to escalate this far. I'm sorry.\n\nBest,\n"
            "Senior Technical Support"
        ),
        "metadata": {"tone": "formal", "sender_role": "customer", "urgency": "high"},
    },

    # ── internal_coordination (6) ─────────────────────────────────────────
    {
        "id": "ic_001",
        "category": "internal_coordination",
        "incoming_email": (
            "Subject: Design review feedback needed by EOD\n\n"
            "Hey team,\n\nThe new dashboard mockups are in Figma (link below). I need "
            "feedback from Engineering and Product by end of day so we can finalize "
            "before sprint planning Monday.\n\nFigma link: [link]\n\nKey questions:\n"
            "1. Is the data table component feasible with our current API?\n"
            "2. Any accessibility concerns with the color palette?\n\n- Design Lead"
        ),
        "sent_reply": (
            "Hey,\n\nJust reviewed the mockups — nice work! Here's my feedback:\n\n"
            "1. Data table: Yes, our current API supports all the columns shown. One note — "
            "the 'real-time status' column would need a WebSocket connection we don't have "
            "yet. Suggest we make that a v2 enhancement and use polling for now.\n\n"
            "2. Accessibility: The light gray text (#B0B0B0) on white backgrounds doesn't "
            "meet WCAG AA contrast ratio (needs 4.5:1, currently ~2.8:1). Recommend "
            "darkening to at least #767676.\n\nOverall the layout and UX flow look solid. "
            "Happy to discuss on a quick call if helpful.\n\n- Engineering"
        ),
        "metadata": {"tone": "casual", "sender_role": "colleague", "urgency": "high"},
    },
    {
        "id": "ic_002",
        "category": "internal_coordination",
        "incoming_email": (
            "Subject: Updated deployment checklist — please review\n\n"
            "Hi all,\n\nI've updated the deployment checklist for our upcoming v2.0 release "
            "(attached). Major changes from v1:\n• Added database migration step\n"
            "• Added rollback procedure\n• New smoke test endpoints\n\nPlease review and "
            "flag any gaps. Deployment is scheduled for Saturday 6am.\n\n- DevOps"
        ),
        "sent_reply": (
            "Hi,\n\nReviewed the checklist — good improvements over v1. A few additions "
            "I'd suggest:\n\n• Pre-deployment: Add a step to verify all feature flags are "
            "set to 'off' for unreleased features\n• Migration: Include expected migration "
            "duration (I estimate 15 min based on staging tests) so the team knows what's "
            "normal vs. concerning\n• Rollback: Add a communication template for notifying "
            "affected users if rollback is needed\n• Post-deployment: Add monitoring dashboard "
            "check (error rates, latency) at T+30min\n\nOtherwise looks comprehensive. I'll "
            "be on-call Saturday if needed.\n\nBest,\nEngineering"
        ),
        "metadata": {"tone": "formal", "sender_role": "colleague", "urgency": "medium"},
    },
    {
        "id": "ic_003",
        "category": "internal_coordination",
        "incoming_email": (
            "Subject: Who owns the analytics pipeline?\n\n"
            "Hey, quick question — I'm trying to debug a data discrepancy in our weekly "
            "metrics report. The numbers in the dashboard don't match what I'm pulling "
            "from the raw database. Who owns the analytics pipeline / ETL process? "
            "I need to understand the transformation logic.\n\n- Product Analyst"
        ),
        "sent_reply": (
            "Hey,\n\nThe analytics pipeline is owned by the Data Engineering team — "
            "specifically, Marcus on the backend and Yuki on the transformation layer.\n\n"
            "Quick context that might help: the dashboard pulls from a materialized view "
            "that refreshes every 6 hours, so if you're comparing against real-time raw "
            "data, there's an expected lag. Also, the ETL applies some deduplication and "
            "session-stitching logic that will make numbers differ from raw counts.\n\n"
            "I'd suggest starting with Marcus — he can walk you through the pipeline "
            "architecture. Slack him at @marcus-data or I can set up a quick call.\n\n- Ops"
        ),
        "metadata": {"tone": "casual", "sender_role": "colleague", "urgency": "medium"},
    },
    {
        "id": "ic_004",
        "category": "internal_coordination",
        "incoming_email": (
            "Subject: Heads up — changing our CI/CD provider\n\n"
            "Team,\n\nWe're switching from Jenkins to GitHub Actions starting August 1st. "
            "This affects all teams that have custom Jenkins pipelines. I'll be holding "
            "migration workshops next week.\n\nAction required: each team needs to audit "
            "their Jenkins configs and identify any custom plugins or scripts that need "
            "porting.\n\n- Platform Engineering"
        ),
        "sent_reply": (
            "Thanks for the heads up!\n\nI've reviewed our team's Jenkins setup. Here's "
            "our audit:\n\n• Standard pipelines (build, test, deploy): 4 pipelines, should "
            "be straightforward to migrate\n• Custom plugins: We use the Slack Notification "
            "plugin and a custom artifact archiver script\n• Secrets: 8 environment secrets "
            "that need to be migrated to GitHub Secrets\n• Cron jobs: 2 nightly builds "
            "triggered via Jenkins cron\n\nI'll attend the migration workshop. Can we get a "
            "staging GitHub Actions environment to test our pipelines before the Aug 1 "
            "cutover?\n\n- Engineering Team Lead"
        ),
        "metadata": {"tone": "casual", "sender_role": "colleague", "urgency": "medium"},
    },
    {
        "id": "ic_005",
        "category": "internal_coordination",
        "incoming_email": (
            "Subject: Budget approval needed for new monitoring tool\n\n"
            "Hi,\n\nI'd like to purchase Datadog for our infrastructure monitoring. Current "
            "state: we're using a patchwork of free tools (UptimeRobot, manual log grep) "
            "that doesn't scale. Cost: $23/host/month × 15 hosts = $345/month.\n\n"
            "Benefits: unified dashboards, alerting, APM, and log management. Expected to "
            "reduce MTTR by 40% based on industry benchmarks.\n\nCan you approve this for "
            "the Q3 budget?\n\n- SRE Lead"
        ),
        "sent_reply": (
            "Hi,\n\nThanks for the thorough proposal. The MTTR improvement alone would "
            "justify the cost given our recent incident frequency.\n\nI'm approving the "
            "$345/month for Datadog in the Q3 budget. A few notes:\n\n1. Please start with "
            "a pilot on our 5 production hosts first (reduces initial commitment to "
            "$115/month) — if results match expectations after 30 days, we'll roll out "
            "to all 15\n2. Set up a shared dashboard for the leadership team so we can "
            "track the MTTR improvement claim\n3. Include Datadog in our vendor review "
            "process — need a signed DPA and security questionnaire\n\nSend me the PO "
            "request and I'll get it processed this week.\n\nBest,\nVP of Engineering"
        ),
        "metadata": {"tone": "formal", "sender_role": "colleague", "urgency": "medium"},
    },
    {
        "id": "ic_006",
        "category": "internal_coordination",
        "incoming_email": (
            "Subject: New intern starting Monday\n\n"
            "Hey, friendly reminder that our summer intern, Casey, starts Monday. They'll "
            "be sitting with the frontend team. Can you make sure the following is ready?\n"
            "• Laptop provisioned with dev environment\n• GitHub/Slack/Jira access\n"
            "• Buddy assigned for first two weeks\n\nThanks!\n- HR"
        ),
        "sent_reply": (
            "Hey!\n\nAll set for Casey's arrival Monday:\n\n✅ Laptop: MacBook Pro provisioned "
            "with our standard dev setup (VS Code, Node 20, Docker)\n✅ Accounts: GitHub "
            "(added to frontend-team), Slack (added to #frontend and #interns), Jira "
            "(contributor access)\n✅ Buddy: Mia from the frontend team has volunteered — "
            "she'll do a welcome walkthrough Monday morning\n\nI've also:\n• Scheduled a "
            "30-min IT onboarding at 10am Monday\n• Added Casey to the Thursday frontend "
            "standup\n• Prepared a 'First Week Guide' doc with useful links and contacts\n\n"
            "Looking forward to having them on the team!\n- Ops"
        ),
        "metadata": {"tone": "casual", "sender_role": "colleague", "urgency": "medium"},
    },
]


def write_dataset(records: list[dict], output_path: Path) -> None:
    """Write records to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅  Wrote {len(records)} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the email/reply dataset")
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        metavar="N",
        help="Generate N additional synthetic pairs via Groq (requires GROQ_API_KEY)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (defaults to backend/data/email_reply_dataset.jsonl)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_path = Path(args.output) if args.output else project_root / "backend" / "data" / "email_reply_dataset.jsonl"

    records = list(SEED_DATA)

    if args.augment > 0:
        print(f"🤖  Augmenting with {args.augment} Groq-generated pairs...")
        try:
            augmented = augment_dataset(args.augment)
            records.extend(augmented)
        except Exception as e:
            print(f"⚠️  Augmentation failed: {e}")
            print("   Falling back to seed data only.")

    write_dataset(records, output_path)


def augment_dataset(n: int) -> list[dict]:
    """
    Generate N additional synthetic email/reply pairs via Groq.

    This is purely a demonstration of scaling the dataset — not required for
    the system to function.
    """
    from groq import Groq
    from backend.app.core.config import get_settings

    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    categories = ["customer_support", "sales_inquiry", "scheduling",
                   "billing_invoice", "complaint_escalation", "internal_coordination"]
    augmented = []

    for i in range(n):
        cat = categories[i % len(categories)]
        prompt = f"""Generate a realistic business email and its professional reply.
Category: {cat}
Return ONLY valid JSON with this exact structure:
{{
  "id": "aug_{i+1:03d}",
  "category": "{cat}",
  "incoming_email": "Subject: ...\n\n...",
  "sent_reply": "...",
  "metadata": {{"tone": "formal|casual", "sender_role": "customer|colleague|vendor", "urgency": "low|medium|high"}}
}}"""
        try:
            resp = client.chat.completions.create(
                model=settings.groq_model_generator,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
            )
            raw = resp.choices[0].message.content.strip()
            # Try to parse JSON from the response
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            record = json.loads(raw)
            augmented.append(record)
            print(f"  Generated {i+1}/{n}: {record['id']}")
        except Exception as e:
            print(f"  ⚠️ Failed to generate pair {i+1}: {e}")

    return augmented


if __name__ == "__main__":
    main()
