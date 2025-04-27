# EDI 850 Purchase Order Implementation Guide

## Overview

The X12 850 transaction set is a Purchase Order document transmitted electronically from a buyer to a supplier. It communicates details about products or services the buyer wants to purchase from the supplier.

## Business Use

The 850 Purchase Order is used to:

- Request products or services from suppliers
- Specify quantities, pricing, and delivery requirements
- Provide shipping and billing information
- Include special handling instructions or other notes

## Document Structure

### Header Section

The header contains information that applies to the entire purchase order:

- Purchase order number
- Purchase order date
- Order type (original, change, etc.)
- Payment terms
- Contract references
- Buyer and seller information

### Detail Section

The detail section includes information about the items being ordered:

- Item identification (part number, UPC, etc.)
- Quantity and unit of measure
- Price information
- Delivery date
- Product/service descriptions
- Item-specific notes or instructions

### Summary Section

The summary section provides control totals:

- Number of line items
- Total monetary amount (optional)
- Control numbers

## Key Segments

| Segment | Name | Description |
|---------|------|-------------|
| BEG | Beginning Segment | Contains PO number, date, and type |
| REF | Reference Information | Provides references to other documents |
| PER | Administrative Contact | Contains contact information |
| DTM | Date/Time Reference | Specifies important dates |
| N1 | Party Identification | Identifies parties involved |
| N2 | Additional Name Information | Additional party name details |
| N3 | Address Information | Street address information |
| N4 | Geographic Location | City, state, zip code |
| PO1 | Purchase Order Line Item | Line item details |
| PID | Product Description | Additional product description |
| CTT | Transaction Totals | Control totals for the transaction |

## Example

```
ST*850*0001
BEG*00*NE*12345678*20220315**20220315
REF*CO*12345
REF*VN*98765
DTM*002*20220330
N1*BY*ACME CORPORATION*92*1234
N3*123 MAIN STREET
N4*ANYTOWN*CA*90001
N1*ST*ACME WAREHOUSE*92*5678
N3*456 DOCK STREET
N4*ANYTOWN*CA*90002
PO1*1*10*EA*49.95**PN*ABC123*VN*987654
PID*F****HIGH QUALITY WIDGET
PO1*2*5*EA*29.95**PN*DEF456*VN*765432
PID*F****STANDARD WIDGET
CTT*2*15
SE*17*0001
```

## Implementation Notes

### Trading Partner Requirements

Before implementing the 850 Purchase Order, discuss these requirements with your trading partner:

1. **Required Segments**: Determine which segments and elements are mandatory, optional, or unused
2. **Qualifiers**: Agree on valid code values and qualifiers
3. **Testing Process**: Establish a testing protocol before moving to production
4. **Acknowledgments**: Decide if 997 Functional Acknowledgments are required
5. **Error Handling**: Determine how errors will be communicated and resolved

### Common Challenges

- **Data Mapping**: Correctly mapping internal data to EDI fields
- **Required Fields**: Ensuring all required fields have valid values
- **Code Values**: Using proper code values as specified by the standard
- **Testing**: Thorough testing with trading partners
- **Compliance**: Meeting trading partner-specific requirements

## Best Practices

1. **Document Specifications**: Create detailed mapping specifications for each trading partner
2. **Validation Rules**: Implement robust validation rules to catch errors early
3. **Testing**: Conduct thorough testing with simulated and real data
4. **Monitoring**: Develop processes to monitor EDI transactions for exceptions
5. **Version Control**: Maintain version control for mapping specifications
6. **Error Handling**: Create clear procedures for error resolution

## Related Documents

- **855 Purchase Order Acknowledgment**: Seller's response to the 850 Purchase Order
- **860 Purchase Order Change Request**: Buyer's request to change an existing PO
- **865 Purchase Order Change Acknowledgment**: Seller's response to the 860
- **870 Order Status Report**: Status update on a purchase order 