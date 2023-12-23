"""
This script houses a toolkit of functions that interact with Interactive Brokers' APIs
to execute trades and pull account data. Some but not all are used in IB_live_trading.py.
It relies on the ib_insync package.
"""


from ib_insync import *
import datetime

def acct_buying_power(ib): 
    buy_pow = float([x.value for x in ib.accountValues() if x.tag == "BuyingPower"][0])
    return buy_pow

def acct_equity_w_loanvalue(ib): 
    equity_w_loanvalue = float([x.value for x in ib.accountValues() if x.tag == "EquityWithLoanValue-S"][0])
    return equity_w_loanvalue

def acct_cash_balance(ib): 
    cash_balance = float([x.value for x in ib.accountValues() if x.tag == "CashBalance"][0])
    return cash_balance

def pull_account_info(ib):
    account_values = ib.accountValues()
    account_info = {}  # Initialize an empty dictionary to store tag-value pairs
    for item in account_values:
        account_info[item.tag] = item.value
    return account_info

def buy_adaptive(ib,tkr, qty, limit_price, urgency = "Urgent"):
    
    contract = Stock(tkr,'SMART' ,currency = 'USD')
    order = Order(
        action = "BUY",                 # Buy order
        orderType = "LMT", 
        totalQuantity = qty,            # Quantity to buy
        lmtPrice = limit_price,               # Limit price
        algoStrategy =  "Adaptive",
        algoParams = [TagValue(tag ='adaptivePriority', value = urgency)] 
    )
    
    trade = ib.placeOrder(contract,order)
    
    return trade, order, contract


def buy_market(ib,tkr, qty):
    # current_time = datetime.now(eastern_tz)
    # expiration_time = (current_time + datetime.timedelta(seconds=20)).strftime('%H:%M:%S')
    
    contract = Stock(tkr,'SMART' ,currency = 'USD')
    order = Order(
        action = "BUY",                 # Buy order
        orderType = "MKT", 
        totalQuantity = qty#,            # Quantity to buy
        # tif = 'GTD',                    # Good 'Til Date duration
        # goodTillDate = expiration_time  # Expiration time
    )
    
    trade = ib.placeOrder(contract,order)
    
    return trade, order, contract


def buy_limit(ib,tkr, qty, limit_price):
    current_time = dt.now(eastern_tz)
    expiration_time = (current_time + datetime.timedelta(seconds=20)).strftime('%H:%M:%S')
    
    contract = Stock(tkr,'SMART' ,currency = 'USD')
    order = Order(
        action = "BUY",                 # Buy order
        orderType = "LMT", 
        lmtPrice = limit_price,   
        totalQuantity = qty,            # Quantity to buy
        tif = 'GTD',                    # Good 'Til Date duration
        goodTillDate = expiration_time  # Expiration time
    )
    
    trade = ib.placeOrder(contract,order)
    
    return trade, order

# buy_limit = buy_limit(tkr, qty, limit_price)

def sell_adaptive(ib,tkr, qty, urgency = "Urgent"):

    #tkr = 'SEDG'
    contract = Stock(tkr, 'SMART', currency = 'USD')
    order = Order(
            action = "SELL",
            orderType = "MKT", 
            totalQuantity = qty, 
            algoStrategy =  "Adaptive",
            algoParams = [TagValue(tag ='adaptivePriority', value = urgency)] 
            )
    
    trade = ib.placeOrder(contract, order)
    
    return trade

def sell_trailing_pct_adaptive(ib,tkr, qty, trail_percent,  urgency = "Urgent"):

    #tkr = 'SEDG'
    contract = Stock(tkr, 'SMART', currency = 'USD')
    order = Order(
            action = "SELL",
            orderType = "TRAIL", 
            totalQuantity = qty,
            trailingPercent = trail_percent,
            algoStrategy =  "Adaptive",
            algoParams = [TagValue(tag ='adaptivePriority', value = urgency)] 
            )
    
    trade = ib.placeOrder(contract,order)
    
    return trade, order


def prof_loss(ib,contract, purchase_price): 
    ticker = ib.reqMktData(contract)
    ib.sleep(1.0)
    market_price = ticker.marketPrice()
    # ib.sleep(1.0)
    prof_loss = market_price / purchase_price
    return prof_loss
    



def trade_to_dict(trade):
    trade_dict = {
        'contract': {
            'symbol': trade.contract.symbol,
            'exchange': trade.contract.exchange,
            'currency': trade.contract.currency
        },
        'order': {
            'orderId': trade.order.orderId,
            'clientId': trade.order.clientId,
            'permId': trade.order.permId,
            'action': trade.order.action,
            'totalQuantity': trade.order.totalQuantity,
            'orderType': trade.order.orderType,
            'lmtPrice': trade.order.lmtPrice,
            'auxPrice': trade.order.auxPrice,
            'algoStrategy': trade.order.algoStrategy,
            'algoParams': {
                param.tag: param.value for param in trade.order.algoParams
            }
        },
        'orderStatus': {
            'orderId': trade.orderStatus.orderId,
            'status': trade.orderStatus.status,
            'filled': trade.orderStatus.filled,
            'remaining': trade.orderStatus.remaining,
            'avgFillPrice': trade.orderStatus.avgFillPrice,
            'permId': trade.orderStatus.permId,
            'parentId': trade.orderStatus.parentId,
            'lastFillPrice': trade.orderStatus.lastFillPrice,
            'clientId': trade.orderStatus.clientId,
            'whyHeld': trade.orderStatus.whyHeld,
            'mktCapPrice': trade.orderStatus.mktCapPrice
        },
        'fills': [
            {
                'contract': {
                    'conId': fill.contract.conId,
                    'symbol': fill.contract.symbol,
                    'exchange': fill.contract.exchange,
                    'currency': fill.contract.currency,
                    'localSymbol': fill.contract.localSymbol,
                    'tradingClass': fill.contract.tradingClass
                },
                'execution': {
                    'execId': fill.execution.execId,
                    'time': fill.execution.time.strftime('%Y-%m-%d %H:%M:%S'),
                    'acctNumber': fill.execution.acctNumber,
                    'exchange': fill.execution.exchange,
                    'side': fill.execution.side,
                    'shares': fill.execution.shares,
                    'price': fill.execution.price,
                    'permId': fill.execution.permId,
                    'clientId': fill.execution.clientId,
                    'orderId': fill.execution.orderId,
                    'liquidation': fill.execution.liquidation,
                    'cumQty': fill.execution.cumQty,
                    'avgPrice': fill.execution.avgPrice,
                    'orderRef': fill.execution.orderRef,
                    'evRule': fill.execution.evRule,
                    'evMultiplier': fill.execution.evMultiplier,
                    'modelCode': fill.execution.modelCode,
                    'lastLiquidity': fill.execution.lastLiquidity
                },
                'commissionReport': {
                    'execId': fill.commissionReport.execId,
                    'commission': fill.commissionReport.commission,
                    'currency': fill.commissionReport.currency,
                    'realizedPNL': fill.commissionReport.realizedPNL,
                    'yield': fill.commissionReport.yield_,
                    'yieldRedemptionDate': fill.commissionReport.yieldRedemptionDate
                },
                'time': fill.time.strftime('%Y-%m-%d %H:%M:%S.%f')
            }
            for fill in trade.fills
        ],
        'log': [
            {
                'time': logEntry.time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'status': logEntry.status,
                'message': logEntry.message,
                'errorCode': logEntry.errorCode
            }
            for logEntry in trade.log
        ],
        'advancedError': trade.advancedError
    }
    return trade_dict