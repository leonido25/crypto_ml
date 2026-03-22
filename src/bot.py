import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv
import os

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

from  predict import get_predictions

@dp.message(Command('start'))
async  def start_handler(message:types.Message):
    ans = await message.answer(('начало работы с ботом'))
    return ans

@dp.message(Command('predict'))
async  def prediction(message:types.Message):
    await message.answer('Считаю прогноз, подожди немного...:')
    res = get_predictions()
    if res is None:
        await  message.answer(('ooops'))
        return
    text = '📊 Прогноз на завтра\n\n'
    for _, row in res.iterrows():
        change = row['predicted price change']
        vol = row['probability of high volatility']
        if change > 0.1:
            emoji = '🟢'
        elif change < -0.1:
            emoji = '🔴'
        else:
            emoji = '🟡'
        warning = ' ⚠️ ' if vol > 0.6 else ''
        text += f'{emoji} {row['Ticker']} {change:+.2f}%   probability of high vol: {vol:.0%}{warning}\n'
    await message.answer(text)

async def main():
    res = await dp.start_polling(bot)
    return res

if __name__ == '__main__':
    asyncio.run(main())