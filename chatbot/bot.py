import logging 
import signal 
import os 
import mysql.connector 
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup 
from telegram.ext import(ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext,)
import ollama 
import time 
from dotenv import load_dotenv 
from telegram.error import Conflict, NetworkError 
from datetime import datetime 

logging.basicConfig(level =logging.INFO, format ='%(asctime)s - %(levelname)s - %(message)s')
logger =logging.getLogger(__name__ )
load_dotenv()

user_data ={}
user_steps ={}
message_lists ={}

def get_db_connection(max_retries =3, delay =2 ):
    for attempt in range(max_retries ):
        try :
            connection =mysql.connector.connect(
            host =os.getenv("MYSQL_HOST","localhost"),
            user =os.getenv("MYSQL_USER"),
            password =os.getenv("MYSQL_PASSWORD"),
            database =os.getenv("MYSQL_DATABASE")
            )
            logger.info("MySQL connection established.")
            return connection 
        except mysql.connector.Error as e :
            logger.error(f"Attempt {attempt + 1}/{max_retries} to connect to MySQL failed: {str(e)}")
            if attempt ==max_retries -1 :
                logger.error("Max retries reached for MySQL connection.")
                return None 
            time.sleep(delay )
    return None 

def init_db(connection ):
    try :
        cursor =connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles(
                user_id BIGINT PRIMARY KEY,
                name VARCHAR(255),
                age INT,
                gender VARCHAR(50),
                skin_tone VARCHAR(50),
                skin_type VARCHAR(50),
                assessment_frequency VARCHAR(50),
                prior_method VARCHAR(50)
            )
        """)
        connection.commit()
        logger.info("Database schema initialized or verified.")
    except mysql.connector.Error as e :
        logger.error(f"Failed to initialize database schema: {str(e)}")
    finally :
        cursor.close()

def show_self_assessment():
    return[
    "🧪 Looks like you've never assessed your skin type before — no worries!\n\n Here are some simple methods to help you figure it out yourself:\n\n"
    "💧 Let’s Help You Discover Your Skin Type!\n\nHere’s a quick and easy 4-step guide you can try at home.No fancy tools needed — just your clean face, some tissue, and good lighting.Let’s go! 👇",
    "🧼 Step 1: The Clean Face Check(aka Wash & Wait)\n\n1.Wash your face with a gentle cleanser and lukewarm water.\n2.Pat it dry with a clean towel.\n3.Don’t apply anything — no moisturizer, sunscreen, or makeup.\n4.Wait for about 2 hours — hands off! 🙈\n5.Now check your skin in a well-lit mirror(natural light is best).",
    "🔍 What to Look For(Step 1):\n\n✨ Oily → Shiny forehead, nose, chin(T-zone)\n🧊 Dry → Skin feels tight, dull, or flaky\n😶‍🌫️ Combination → T-zone is shiny, cheeks feel dry/normal\n🌿 Normal → Skin feels comfy — not oily or dry",
    "📄 Step 2: The Tissue Trick(Blotting Test)\n\n1.After a few hours(or midday), grab a tissue or clean toilet paper.\n2.Gently press it on your forehead, nose, chin, and cheeks.\n3.Hold it to light and see what shows up!",
    "🔍 What to Look For(Step 2):\n\n✨ Oily → Tissue has oil spots from most areas\n🧊 Dry → Tissue stays clean, skin feels tight\n😶‍🌫️ Combination → Oil only on T-zone, not cheeks\n🌿 Normal → Little to no oil, skin feels balanced",
    "🔎 Step 3: Pore Patrol(Mirror Time!)\n\n1.About 1 hour after washing your face, look closely in the mirror.\n2.Focus on the size of your pores and how your skin feels.",
    "🔍 What to Look For(Step 3):\n\n✨ Oily → Large pores on T-zone, skin feels greasy\n🧊 Dry → Tiny pores, skin feels rough or flaky\n😶‍🌫️ Combination → Larger pores on T-zone, smaller ones on cheeks\n🌿 Normal → Small–medium pores, skin feels smooth",
    "🌡️ Step 4: Is Your Skin Sensitive?\n\nOver a few days, observe how your skin reacts to:\n- Weather(sun, wind, cold)\n- New gentle skincare products\n- Touching or rubbing your face",
    "🔍 What to Look For(Step 4):\n\n🚨 Sensitive → Skin gets red, itchy, or burns easily\n✅ Not Sensitive → No big reactions unless product is very harsh",
    "✅ Pro Tips for Best Results:\n\n- Try tests on clean, makeup-free skin\n- Repeat 2–3 times during the week(skin can change!)\n- Use natural light when checking\n- Stay hydrated and eat normally — it affects your skin too!",
    "🧴 So, What’s Your Skin Type?\n\n✨ Oily → Shiny face, big pores, prone to pimples\n🧊 Dry → Tight or flaky skin, small pores, no shine\n😶‍🌫️ Combination → Oily T-zone + dry or normal cheeks\n🌿 Normal → Smooth, balanced, not too oily or dry\n\nLet me know which one sounds like you!"
    ]

skin_tips ={
"Oily":(
"Focus on non-comedogenic, oil-free products.Use a gentle foaming cleanser twice daily "
"to remove excess sebum without disrupting the skin barrier.Incorporate ingredients like salicylic acid or niacinamide "
"to help regulate oil production and minimize pores."
),
"Dry":(
"Use a creamy, hydrating cleanser and follow with a thick moisturizer containing ceramides or hyaluronic acid."
"Avoid long, hot showers and harsh exfoliants.Consider using a humidifier in dry environments to maintain skin hydration."
),
"Combination":(
"Tailor your routine to different zones of your face.Use a gentle, pH-balanced cleanser and opt for lightweight hydration on the T-zone, "
"while applying richer moisturizers to drier areas like the cheeks.Exfoliate with mild AHAs to balance oil and texture."
),
"Normal":(
"Maintain your skin’s balance with a consistent routine: a mild cleanser, daily SPF, and a moisturizer suited for your climate."
"Avoid over-exfoliating and stay hydrated to support your natural skin barrier."
)
}

def check_ollama_connection(max_retries =3, delay =2 ):
    for attempt in range(max_retries ):
        try :
            ollama.list()
            logger.info("Ollama connection established.")
            return True 
        except Exception as e :
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to connect to Ollama failed: {str(e)}")
            if attempt ==max_retries -1 :
                logger.error("Max retries reached for Ollama connection.")
                return False 
            time.sleep(delay )
    return False 

def get_response(user_id, prompt ):
    logger.info(f"Getting response for user {user_id}: {prompt}")
    message_lists[user_id ].append({"role":"user","content":prompt })
    for attempt in range(3 ):
        try :
            response =ollama.chat(
            model ="llama3.1:8b",
            messages =message_lists[user_id ]
            )
            assistant_response =response["message"]["content"]
            message_lists[user_id ].append({"role":"assistant","content":assistant_response })
            logger.info(f"Response received: {assistant_response}")
            return assistant_response 
        except Exception as e :
            logger.error(f"Attempt {attempt + 1}/3 to get response from Ollama failed: {str(e)}")
            if attempt ==2 :
                return "Sorry, I'm having trouble connecting to the AI model.Please try again later."
            time.sleep(2 )
    return None 

def create_keyboard(options, prefix ):
    try :
        keyboard =[[InlineKeyboardButton(option, callback_data =f"{prefix}_{option.lower()}")]for option in options ]
        logger.info(f"Created keyboard for {prefix}: {options}")
        return InlineKeyboardMarkup(keyboard )
    except Exception as e :
        logger.error(f"Failed to create keyboard for {prefix}: {str(e)}")
        return None 

def is_skin_related(text ):
    skin_keywords =["skin","acne","moisturizer","cleanser","dryness","oil","pores","pimples","blackheads","skincare"]
    return any(keyword in text.lower() for keyword in skin_keywords )

def cleanup():
    logger.info("Cleaning up user data and profiles...")
    user_data.clear()
    user_steps.clear()
    message_lists.clear()
    connection =get_db_connection(max_retries =2 )
    if connection :
        try :
            cursor =connection.cursor()
            cursor.execute("DELETE FROM user_profiles")
            connection.commit()
            logger.info("Deleted all records from user_profiles table.")
        except Exception as e :
            logger.error(f"Failed to delete database records: {str(e)}")
        finally :
            cursor.close()
            connection.close()
    else :
        logger.error("No database connection available for cleanup.")
    logger.info("Cleanup completed.")

async def start(update :Update, context :CallbackContext ):
    user_id =update.effective_user.id 
    user_data[user_id ]={}
    user_steps[user_id ]="name"
    message_lists[user_id ]=[
    {"role":"system","content":"Hello, this is Dora AI.Here to help you understand your skin better!"},
    {"role":"assistant","content":"What is your name?"}
    ]
    await update.message.reply_text("👋 Hello! Welcome to Dora AI — your personal skin health assistant.")
    await update.message.reply_text("📝 We’ll go through a short questionnaire to help ensure a more accurate and personalized diagnosis.Please answer honestly — and don’t worry if you're unsure about anything.We're here to guide you! 😊")
    await update.message.reply_text("🔁 FYI => If you made a mistake or want to start over, you can type /restart to begin the questionnaire again.")
    await update.message.reply_text("🙌 Great! Let’s get started.\nWhat’s your name?")

async def handle_message(update :Update, context :CallbackContext ):
    user_id =update.effective_user.id 
    text =update.message.text.strip()
    logger.info(f"Received message from user {user_id} at step {user_steps.get(user_id, 'none')}: {text}")

    if user_id not in user_steps :
        await update.message.reply_text("Please start the questionnaire with /start.")
        return 

    step =user_steps[user_id ]

    if step =="name":
        if text :
            user_data[user_id ]["name"]=text 
            message_lists[user_id ].append({"role":"user","content":text })
            message_lists[user_id ].append({"role":"assistant","content":f"Nice to meet you, {text}! What is your age?"})
            user_steps[user_id ]="age"
            await update.message.reply_text(f"😊 Nice to meet you, {text}! How old are you?")
        else :
            await update.message.reply_text("Please enter a valid name.")

    elif step =="age":
        try :
            age =int(text )
            if 13 <=age <=90 :
                user_data[user_id ]["age"]=age 
                message_lists[user_id ].append({"role":"user","content":text })
                user_steps[user_id ]="gender"
                keyboard =create_keyboard(["Male","Female"],"gender")
                if keyboard :
                    await update.message.reply_text("What is your gender?", reply_markup =keyboard )
                else :
                    logger.error("Failed to create gender keyboard")
                    await update.message.reply_text("Sorry, something went wrong.Please try again with /start.")
            else :
                await update.message.reply_text("Oops! You need to be 13 or older to access Dora Chatbot’s features.Please come back when you're a bit older! 😊")
        except ValueError :
            await update.message.reply_text("❗ That doesn't look like a valid age.Please enter a number.")

    elif step =="complete":
        if is_skin_related(text ):
            response =get_response(user_id, text )
            await update.message.reply_text(response if response else "Sorry, I'm having trouble processing your request.Please try again.")
    else :
        await update.message.reply_text("Please select an option from the buttons provided below.")

async def handle_callback(update :Update, context :CallbackContext ):
    query =update.callback_query 
    user_id =query.from_user.id 
    data =query.data 
    await query.answer()
    logger.info(f"Received callback from user {user_id}: {data}")

    if user_id not in user_steps :
        logger.warning(f"User {user_id} not in user_steps")
        await query.message.reply_text("🚀 Let’s get started! Please type /start to begin the questionnaire.")
        return 

    step =user_steps[user_id ]
    logger.info(f"Processing callback for user {user_id} at step {step}")

    try :
        logger.debug(f"Parsing callback data: {data}")
        parts =data.rsplit("_", 1 )
        if len(parts )!=2 :
            logger.error(f"Invalid callback data format: {data}, expected prefix_value")
            await query.message.reply_text("Something went wrong.Please select an option again.")
            return 
        prefix, value =parts 
        logger.debug(f"Parsed prefix: {prefix}, value: {value}")
    except Exception as e :
        logger.error(f"Error parsing callback data: {str(e)}")
        await query.message.reply_text("An error occurred while processing your selection.Please try again.")
        return 

    if step =="gender"and prefix =="gender":
        if value in["male","female"]:
            user_data[user_id ]["gender"]=value.capitalize()
            message_lists[user_id ].append({"role":"user","content":value.capitalize()})
            user_steps[user_id ]="skin_tone"
            keyboard =create_keyboard(["Fair","Medium","Dark","Other"],"skin_tone")
            if keyboard :
                    await query.message.reply_text("🧑‍🎨 Please select the option that best describes your skin tone.", reply_markup =keyboard )
            else :
                    logger.error("Failed to create skin tone keyboard")
                    await query.message.reply_text("⚠️ Oops, something went wrong....!")
                    await query.message.reply_text("Would you like to restart the questionnaire?")
        else :
                logger.error(f"Invalid gender value: {value}")
                await query.message.reply_text(
                "❌ That doesn't seem like a valid answer.\nPlease choose from the available options above.😊"
                )

    elif step =="skin_tone"and prefix =="skin_tone":
            if value in["fair","medium","dark","other"]:
                user_data[user_id ]["skin_tone"]=value.capitalize()
                message_lists[user_id ].append({"role":"user","content":value.capitalize()})
                user_steps[user_id ]="skin_type"

                keyboard =create_keyboard(
["Oily","Dry","Combination","Normal","Not Sure"],"skin_type"
                )

                if keyboard :
                    await query.message.reply_text(
                    "🎉 Thanks for sharing your skin tone!\nNow let’s move on 👇\nWhat best describes your *skin type*?",
                    reply_markup =keyboard 
                    )
                else :
                    logger.error("Failed to create skin type keyboard")
                    await query.message.reply_text(
                    "⚠️ Hmm...something went wrong while preparing the next step.\nPlease try again by typing /start."
                    )
            else :
                logger.error(f"Invalid skin tone value: {value}")
                await query.message.reply_text(
                "❌ That doesn't seem like a valid answer.\nPlease choose from the available options above.😊"
                )
    elif step =="skin_type"and prefix =="skin_type":
            if value in["oily","dry","combination","normal","not sure"]:
                user_data[user_id ]["skin_type"]=value.capitalize()
                message_lists[user_id ].append({"role":"user","content":value.capitalize()})
                user_steps[user_id ]="assessment_frequency"

                keyboard =create_keyboard(["Weekly","Monthly","Rarely","Never"],"assessment_frequency")

                if keyboard :
                    await query.message.reply_text(
                    "🕵️‍♀️ Got it! How often do you usually check your skin health?",
                    reply_markup =keyboard 
                    )
                else :
                    logger.error("Failed to create assessment frequency keyboard")
                    await query.message.reply_text(
                    "⚠️ Hmm...something went wrong while preparing the next step.\nPlease try again by typing /start."
                    )
            else :
                logger.error(f"Invalid skin type value: {value}")
                await query.message.reply_text(
                "❌ That doesn't seem like a valid answer.\nPlease choose from the available options above.😊"
                )

    elif step =="assessment_frequency"and prefix =="assessment_frequency":
            if value in["weekly","monthly","rarely","never"]:
                user_data[user_id ]["assessment_frequency"]=value.capitalize()
                message_lists[user_id ].append({"role":"user","content":value.capitalize()})
                user_steps[user_id ]="prior_method"

                keyboard =create_keyboard(
["Dermatologist","Self-assessment","Other apps","No assessment"],
                "prior_method"
                )

                if keyboard :
                    await query.message.reply_text(
                    "🧑‍⚕️ Almost there!\nBefore using this tool, how did you usually identify your skin type?",
                    reply_markup =keyboard 
                    )
                else :
                    logger.error("Failed to create prior method keyboard")
                    await query.message.reply_text(
                    "⚠️ Hmm...something went wrong while preparing the next step.\nPlease try again by typing /start."
                    )
            else :
                logger.error(f"Invalid assessment frequency value: {value}")
                await query.message.reply_text(
                "❌ That doesn't seem like a valid answer.\nPlease choose from the available options above.😊"
                )

    elif step =="prior_method"and prefix =="prior_method":
        if value in["dermatologist","self-assessment","other apps","no assessment"]:
         user_data[user_id ]["prior_method"]=value.capitalize()
         message_lists[user_id ].append({"role":"user","content":value.capitalize()})

        if value =="no assessment":
            for msg in show_self_assessment():
                await query.message.reply_text(msg )

            user_steps[user_id ]="skin_type"
            keyboard =create_keyboard(["Oily","Dry","Combination","Normal","Not Sure"],"skin_type")
            await query.message.reply_text(
            "🔄 Now that you've learned how to assess your skin — how would you describe your *skin type*?",
            reply_markup =keyboard 
            )
            return 
        else :
            skin_type =user_data[user_id ]["skin_type"]
            tip =skin_tips.get(skin_type,"It's always a good idea to consult a dermatologist for deeper insight.")
            response =(
            "✅ Thanks for completing the questionnaire!\n\n"
            f"Since you have *{skin_type}* skin, here are some tips just for you:\n\n"
            f"{tip}\n"
            )

            message_lists[user_id ].append({"role":"assistant","content":response })
            user_steps[user_id ]="complete"

            await query.message.reply_text(response )
            await query.message.reply_text("💾 If you're done, type /done to save your profile.")
    else :
        logger.error(f"Invalid prior method value: {value}")
        await query.message.reply_text(
        "❌ That doesn't seem like a valid answer.\nPlease choose from the available options above.😊"
        )

async def save_profile(update :Update, context :CallbackContext ):
    user_id =update.effective_user.id 
    logger.info(f"Attempting to save profile for user {user_id}")

    logger.debug(f"user_data for {user_id}: {user_data.get(user_id, {})}")

    if user_id not in user_data or not user_data[user_id ]:
        logger.warning(f"No profile data available for user {user_id}")
        await update.message.reply_text("⚠️ It looks like you haven’t completed the questionnaire yet.\nPlease finish it first so we can save your profile.😊")
        return 

    required_fields ={"name","age","gender","skin_tone","skin_type","assessment_frequency","prior_method"}
    missing_fields =required_fields -set(user_data[user_id ].keys())
    if missing_fields :
        logger.error(f"Missing required fields for user {user_id}: {missing_fields}")
        await update.message.reply_text(f"⚠️ Oops! Some info is missing: {', '.join(missing_fields)}.\nPlease complete the questionnaire to save your profile.")
        return 

    for field in["name","gender","skin_tone","skin_type","assessment_frequency","prior_method"]:
        if len(str(user_data[user_id ][field ]))>50 :
            logger.error(f"Field {field} for user {user_id} exceeds 50 characters: {user_data[user_id][field]}")
            await update.message.reply_text(f"⚠️ The field '{field.capitalize()}' is too long.Please keep it under 50 characters.")
            return 

    connection =get_db_connection(max_retries =2 )
    if not connection :
        logger.error(f"Failed to establish MySQL connection for user {user_id}")
        await update.message.reply_text("❌ Unable to save your profile right now due to a database issue.\nPlease try again in a bit!")
        return 

    try :
        init_db(connection )
        cursor =connection.cursor()

        data =(
        user_id,
        user_data[user_id ]["name"],
        user_data[user_id ]["age"],
        user_data[user_id ]["gender"],
        user_data[user_id ]["skin_tone"],
        user_data[user_id ]["skin_type"],
        user_data[user_id ]["assessment_frequency"],
        user_data[user_id ]["prior_method"]
        )

        cursor.execute("SELECT user_id FROM user_profiles WHERE user_id = %s",(user_id,))
        exists =cursor.fetchone()

        logger.debug(f"Executing query for user {user_id}.Exists: {bool(exists)}, Data: {data}")

        if exists :
            logger.info(f"Updating existing profile for user {user_id}")
            query ="""
                UPDATE user_profiles 
                SET name = %s, age = %s, gender = %s, skin_tone = %s, skin_type = %s, 
                    assessment_frequency = %s, prior_method = %s
                WHERE user_id = %s
            """
            cursor.execute(query,(data[1 ], data[2 ], data[3 ], data[4 ], data[5 ], data[6 ], data[7 ], data[0 ]))
        else :
            logger.info(f"Inserting new profile for user {user_id}")
            query ="""
                INSERT INTO user_profiles(user_id, name, age, gender, skin_tone, skin_type, assessment_frequency, prior_method)
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, data )

        connection.commit()
        logger.info(f"Profile saved successfully for user {user_id}")
        await update.message.reply_text("✅ All done! Your profile has been saved successfully.💾\n\nThanks for your time! You can now ask me anything about skincare 💬")

        cursor.execute("SELECT * FROM user_profiles WHERE user_id = %s",(user_id,))
        result =cursor.fetchone()
        logger.debug(f"Database content for user {user_id}: {result}")

    except mysql.connector.Error as e :
        logger.error(f"Database error while saving profile for user {user_id}: {str(e)}")
        await update.message.reply_text(f"❌ Failed to save your profile due to a database issue:\n{str(e)}")
    except Exception as e :
        logger.error(f"Unexpected error while saving profile for user {user_id}: {str(e)}")
        await update.message.reply_text(f"⚠️ Something unexpected happened:\n{str(e)}\nPlease try again.")
    finally :
        cursor.close()
        connection.close()

async def restart(update :Update, context :CallbackContext ):
    user_id =update.effective_user.id 
    user_data[user_id ]={}
    user_steps[user_id ]="name"
    message_lists[user_id ]=[
    {"role":"system","content":"Hello, this is Dora AI.Here to help you understand your skin better!"},
    {"role":"assistant","content":"What is your name?"}
    ]
    await update.message.reply_text("🔄 Your questionnaire has been restarted as requested!")
    await update.message.reply_text("👋 Hello! Welcome to Dora AI — your personal skin health assistant.")
    await update.message.reply_text("📝 We’ll go through a short questionnaire to help ensure a more accurate and personalized diagnosis.Please answer honestly — and don’t worry if you're unsure about anything.We're here to guide you! 😊")
    await update.message.reply_text("🔁 FYI => If you made a mistake or want to start over, you can type /restart to begin the questionnaire again.")
    await update.message.reply_text("🙌 Great! Let’s get started.\nWhat’s your name?")

async def error_handler(update :Update, context :CallbackContext ):
    logger.error(f"Update {update} caused error: {context.error}")
    if isinstance(context.error, Conflict ):
        logger.error("Conflict error: Multiple bot instances detected.")
        await update.message.reply_text(
        "⚠️ Oops! Another session of this bot is currently active.\nPlease close other instances and try again with /start."
        ) if update else logger.info("No update available to send conflict error message.")
    elif isinstance(context.error, NetworkError ):
        logger.error("Network error occurred.Retrying connection...")
        await update.message.reply_text(
        "📡 Network issue detected.\nPlease check your internet connection and try again shortly."
        ) if update else logger.info("No update available for network error message.")

def main():
    # Removed sensitive token printing

    if not check_ollama_connection():
        logger.error("Cannot start bot: Ollama connection failed.Proceeding without AI responses.")
    bot_token =os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token :
        logger.error("TELEGRAM_BOT_TOKEN not set.")
        return 
    app =ApplicationBuilder().token(bot_token ).build()

    app.add_handler(CommandHandler("start", start ))
    app.add_handler(CommandHandler("save", save_profile ))
    app.add_handler(CommandHandler("done", save_profile ))
    app.add_handler(CommandHandler("restart", restart ))
    app.add_handler(MessageHandler(filters.TEXT &~filters.COMMAND, handle_message ))
    app.add_handler(CallbackQueryHandler(handle_callback ))
    app.add_error_handler(error_handler )

    def handle_shutdown(signum, frame ):
        logger.info("🛑 Shutdown signal received.Cleaning up resources...")
        cleanup()
        app.stop()
        logger.info("✅ Bot stopped gracefully.")
        exit(0 )

    signal.signal(signal.SIGINT, handle_shutdown )
    signal.signal(signal.SIGTERM, handle_shutdown )

    start_time =datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"🚀 Starting Telegram bot at {start_time} EEST")
    try :
        app.run_polling(allowed_updates =Update.ALL_TYPES )
    except(Conflict, NetworkError ) as e :
        logger.error(f"Critical error on startup: {str(e)}")
        print(f"❌ Startup Error: {str(e)}\n➡️ Please ensure no other instance is running and your network is stable.")
        return 
    except Exception as e :
        logger.error(f"Unexpected error on startup: {str(e)}")
        print(f"❗ Unexpected error occurred: {str(e)}\nCheck logs for more details.")
        return 

if __name__ =="__main__":
    main()