# bot.py
import os
import time

from scraper import get_updates

import discord
from dotenv import load_dotenv
# from discord.ext import tasks

print("Initialization")

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

client = discord.Client(intents=discord.Intents.default())

@client.event
async def on_ready():
    print("Ready!")
    message_channel = client.get_channel(693453766120177667)

    new_greetings = get_updates()

    for greeting in new_greetings:
        await message_channel.send(greeting)
        time.sleep(5)

    await client.close()

client.run(TOKEN)
exit()
