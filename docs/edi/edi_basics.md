# Electronic Data Interchange (EDI) Basics

## What is EDI?

Electronic Data Interchange (EDI) is the computer-to-computer exchange of business documents in a standard electronic format between business partners. EDI replaces paper-based documents such as purchase orders, invoices, and shipping notices with electronic equivalents.

## Benefits of EDI

- **Speed**: Transactions that took days by mail can be completed in minutes via EDI
- **Accuracy**: Reduces errors by eliminating manual data entry and handling
- **Cost**: Reduces costs associated with paper, printing, storage, and postage
- **Efficiency**: Streamlines business processes and automates routine tasks
- **Security**: Provides secure document transmission with encryption and access controls

## Common EDI Standards

### X12

Developed by ANSI (American National Standards Institute), X12 is predominantly used in North America. It includes standards for various business documents such as:

- 810: Invoice
- 850: Purchase Order
- 856: Advanced Shipping Notice
- 997: Functional Acknowledgment

### EDIFACT

Developed by the United Nations, EDIFACT (Electronic Data Interchange for Administration, Commerce and Transport) is used internationally. Common EDIFACT messages include:

- ORDERS: Purchase Order
- INVOIC: Invoice
- DESADV: Dispatch Advice (similar to X12 856)
- CONTRL: Acknowledgment Message

## EDI Document Structure

### X12 Structure

A typical X12 transaction consists of:

1. **Interchange Control Header (ISA)**: Identifies the sender and receiver
2. **Functional Group Header (GS)**: Groups related transaction sets
3. **Transaction Set Header (ST)**: Marks the beginning of a transaction
4. **Data Segments**: Contains the actual business data
5. **Transaction Set Trailer (SE)**: Marks the end of a transaction
6. **Functional Group Trailer (GE)**: Marks the end of a functional group
7. **Interchange Control Trailer (IEA)**: Marks the end of the interchange

Example of X12 850 Purchase Order (simplified):

```
ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       *190101*1200*U*00401*000000001*0*T*>
GS*PO*SENDER*RECEIVER*20190101*1200*1*X*004010
ST*850*0001
BEG*00*SA*123456**20190101
N1*BY*Buying Company
N1*ST*Shipping Address
PO1*1*10*EA*9.99**PN*ITEM123
CTT*1
SE*7*0001
GE*1*1
IEA*1*000000001
```

## Implementation Considerations

- **Trading Partner Agreements**: Document the EDI requirements between partners
- **EDI Translation Software**: Converts internal data formats to/from EDI standards
- **Communication Methods**: AS2, SFTP, VAN, or API-based transmission
- **Testing**: Required before production implementation
- **Monitoring**: Regular monitoring of EDI flows for errors or exceptions

## Resources for Learning More

- ANSI X12 Committee: [www.x12.org](http://www.x12.org)
- UN/EDIFACT: [www.unece.org/trade/untdid/welcome.html](http://www.unece.org/trade/untdid/welcome.html)
- GS1: [www.gs1.org](http://www.gs1.org) 