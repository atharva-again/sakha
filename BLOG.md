# Sakha — A Real Friend

## Introduction

Imagine someone close to you is in a hospital ward. It could be an ICU or a general ward, but the situation is almost always the same. There are around 17 to 18 patients in a single ward, and only one or two attendants managing all of them. They are responsible for everything — giving medicines on time, checking BP, temperature, oxygen levels, managing routines, and responding when doctors call. Every patient needs attention, every task has a deadline.

Now think about this: your family member is just one of those 18 patients. The attendant responsible is constantly moving from bed to bed, patient to patient. At the same time, family members are calling them from different directions, asking for help. This continues for hours, through an entire 8-hour shift, and slowly it becomes overwhelming. Not because they don’t care, but because one person simply cannot manage 18 lives perfectly at the same time.

On top of this, there is another major problem — the shift change. In the morning, one attendant handles the ward, and in the evening, a new one takes over. The handover is often incomplete or rushed, and the new attendant ends up managing the same 18 patients with only partial information.

This is the reality in many hospitals today, and this is the problem we are solving.

We are building **Sakha**

> *Sakha is an AI agent that can manage wards. It has the ability to take the whole state parameters of the ward at any time and provide the most optimal action to take.*

---

## System Overview

So we asked a simple question:

> *What if there was something that could quietly keep track of everything?*

Sakha works as a **time-aware assistant** that continuously watches over tasks, schedules, and patient states.

It takes in information about patients, processes what needs to be done, and then reminds or alerts caregivers at the right time.

You can think of it as something that:

- Never forgets  
- Never gets distracted  
- Always knows what matters most right now  

---

## Environment Design

At its core, Sakha treats each patient like a **living, changing state**.

Every patient has:

- Current vitals  
- A schedule of medications  
- A list of pending tasks  

Time keeps moving, and Sakha keeps checking:

- What needs to be done next?  
- What is overdue?  
- What suddenly became urgent?  

It then **reorders priorities dynamically**.

If something critical happens, like abnormal vitals, it immediately jumps to the top.

And every time an action is completed (or missed), Sakha updates its understanding of the situation.

This creates a system that is **always aware, always adapting**.

---

## Reward System

Now, to make Sakha not just track tasks but also *understand good behavior*, we introduced a **reward system**.

Think of it like feedback:

- When something is done on time → positive reward  
- When something is delayed or missed → penalty  

For example:

- Giving medicine on time → reward  
- Missing a critical alert → penalty  
- Detecting an issue early → higher reward  

Over time, this helps the system understand:

> *What actions actually matter the most?*

It’s not just about doing tasks—it’s about doing the **right tasks at the right time**.

The reward system also considers:

- Timing (faster is better)  
- Priority (critical tasks matter more)  
- Consistency over time  

Instead of training one big complex model, this reward mechanism acts like a **guiding signal**, helping Sakha behave more intelligently and reliably.

---

## How the Environment Works (Execution Flow)

Here’s what happens behind the scenes:

1. Patients are initialized with their data  
2. Tasks and schedules are registered  
3. Time keeps moving continuously  
4. Sakha checks what needs attention  
5. Tasks are prioritized dynamically  
6. Alerts are sent when needed  
7. Everything gets logged and updated  

This loop keeps running—quietly making sure nothing important is missed.

---

## Results

When we tested Sakha in simulated environments, we saw clear improvements:

- Fewer missed medications  
- More consistent monitoring of vitals  
- Better organization of tasks  

But more importantly, it changed how the system *felt*:

Instead of reacting to problems, the workflow became **structured and proactive**.

---

## Benefits

Sakha helps in very practical ways:

**Reliability:** Important tasks are not forgotten.

**Reduced Stress:** Caregivers don’t have to remember everything.  

**Better Focus:** Attention goes where it’s needed most.

**Scalability:** More patients can be handled without overload. 

---

## Future Scope

Right now, Sakha works as a single intelligent assistant helping manage multiple patients. But real hospital scenarios can get even more complex (multiple wards, different esclation, ICUs, OTs).

The next step is a **multi-agent system**, where:

- Each agent focuses on a subset of patients  
- Agents coordinate with each other  
- Critical situations are handled in parallel  

This would allow Sakha to:

- Respond faster during simultaneous emergencies  
- Distribute attention intelligently  
- Maintain stability under extreme workload  

Looking further ahead, as healthcare moves toward automation, we can imagine a future where **robots assist as nurses or attendants**.

In such a system, Sakha can act as the **intelligence layer behind physical agents**, guiding:

- What action to take  
- Which patient needs priority  
- How to coordinate multiple robotic units  

At that point, Sakha evolves from a digital assistant to a **coordinating brain for a fully assisted care system**.

---

## Conclusion

At its heart, Sakha is not just a system.

It’s a **support system**.

A quiet assistant that stays alert when humans are tired.  
A guide that helps in moments of pressure.  
A real friend when things get difficult.  

And that’s the goal:

> **To make sure that even in the busiest, most stressful environments, care never breaks down.**
