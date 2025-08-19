import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Seeding stock data...');

  // 1. 시장 지수 데이터
  const marketIndices = [
    { symbol: 'KOSPI', name: '코스피', value: 2501.23, change: 15.67, changePercent: 0.63 },
    { symbol: 'KOSDAQ', name: '코스닥', value: 698.45, change: -8.21, changePercent: -1.16 },
    { symbol: 'KOSPI200', name: '코스피200', value: 321.15, change: 2.34, changePercent: 0.73 },
    { symbol: 'DJI', name: '다우존스', value: 34721.12, change: 182.01, changePercent: 0.53 },
    { symbol: 'NASDAQ', name: '나스닥', value: 13711.00, change: -123.45, changePercent: -0.89 },
    { symbol: 'S&P500', name: 'S&P 500', value: 4488.28, change: 23.17, changePercent: 0.52 },
  ];

  for (const index of marketIndices) {
    await prisma.stock_MarketIndex.upsert({
      where: { symbol: index.symbol },
      update: {
        value: index.value,
        change: index.change,
        changePercent: index.changePercent,
      },
      create: index,
    });
  }
  console.log('✅ Market indices created');

  // 2. 종목 마스터 데이터
  const stockSymbols = [
    { symbol: '005930', nameKr: '삼성전자', market: 'KOSPI', sector: '반도체', marketCap: BigInt('408500000000000') },
    { symbol: '000660', nameKr: 'SK하이닉스', market: 'KOSPI', sector: '반도체', marketCap: BigInt('75000000000000') },
    { symbol: '035420', nameKr: 'NAVER', market: 'KOSPI', sector: 'IT', marketCap: BigInt('45000000000000') },
    { symbol: '035720', nameKr: '카카오', market: 'KOSPI', sector: 'IT', marketCap: BigInt('25000000000000') },
    { symbol: '207940', nameKr: '삼성바이오로직스', market: 'KOSPI', sector: '바이오', marketCap: BigInt('55000000000000') },
    { symbol: '068270', nameKr: '셀트리온', market: 'KOSPI', sector: '바이오', marketCap: BigInt('23700000000000') },
    { symbol: '005380', nameKr: '현대차', market: 'KOSPI', sector: '자동차', marketCap: BigInt('38000000000000') },
    { symbol: '051910', nameKr: 'LG화학', market: 'KOSPI', sector: '화학', marketCap: BigInt('28000000000000') },
    { symbol: '006400', nameKr: '삼성SDI', market: 'KOSPI', sector: '2차전지', marketCap: BigInt('32000000000000') },
    { symbol: '373220', nameKr: 'LG에너지솔루션', market: 'KOSPI', sector: '2차전지', marketCap: BigInt('99800000000000') },
    { symbol: '105560', nameKr: 'KB금융', market: 'KOSPI', sector: '금융', marketCap: BigInt('21000000000000') },
    { symbol: '055550', nameKr: '신한지주', market: 'KOSPI', sector: '금융', marketCap: BigInt('19000000000000') },
    { symbol: '000270', nameKr: '기아', market: 'KOSPI', sector: '자동차', marketCap: BigInt('33500000000000') },
    { symbol: '012330', nameKr: '현대모비스', market: 'KOSPI', sector: '자동차부품', marketCap: BigInt('20300000000000') },
    { symbol: '028260', nameKr: '삼성물산', market: 'KOSPI', sector: '종합상사', marketCap: BigInt('18000000000000') },
  ];

  for (const stock of stockSymbols) {
    const created = await prisma.stock_Symbol.upsert({
      where: { symbol: stock.symbol },
      update: {},
      create: stock,
    });

    // 3. 시세 데이터 추가
    const priceData = {
      '005930': { close: 68500, change: 2100, changePercent: 3.16, volume: BigInt('15234567') },
      '000660': { close: 115000, change: 3500, changePercent: 3.14, volume: BigInt('8901234') },
      '035420': { close: 215000, change: 6000, changePercent: 2.87, volume: BigInt('1234567') },
      '035720': { close: 45200, change: 1200, changePercent: 2.73, volume: BigInt('3456789') },
      '207940': { close: 785000, change: 20000, changePercent: 2.61, volume: BigInt('234567') },
      '068270': { close: 172000, change: 3000, changePercent: 1.77, volume: BigInt('8765432') },
      '005380': { close: 185000, change: -5500, changePercent: -2.89, volume: BigInt('2345678') },
      '051910': { close: 412000, change: -11000, changePercent: -2.60, volume: BigInt('567890') },
      '006400': { close: 398000, change: -9500, changePercent: -2.33, volume: BigInt('345678') },
      '373220': { close: 425000, change: -5000, changePercent: -1.16, volume: BigInt('12345678') },
      '105560': { close: 52100, change: -1100, changePercent: -2.07, volume: BigInt('1234567') },
      '055550': { close: 58900, change: -300, changePercent: -0.52, volume: BigInt('987654') },
      '000270': { close: 82500, change: 1500, changePercent: 1.85, volume: BigInt('9876543') },
      '012330': { close: 215000, change: -3000, changePercent: -1.38, volume: BigInt('7654321') },
      '028260': { close: 102000, change: -2300, changePercent: -2.21, volume: BigInt('890123') },
    };

    if (priceData[stock.symbol]) {
      const price = priceData[stock.symbol];
      await prisma.stock_Quote.create({
        data: {
          stockId: created.id,
          open: price.close - price.change,
          high: price.close * 1.01,
          low: price.close * 0.99,
          close: price.close,
          change: price.change,
          changePercent: price.changePercent,
          volume: price.volume,
        },
      });
    }
  }
  console.log('✅ Stock symbols and quotes created');

  console.log('🎉 Seed completed successfully!');
}

main()
  .catch((e) => {
    console.error('Error seeding data:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });