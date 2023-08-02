#ifndef	__UART_H__
#define __UART_H__

#include "stdint.h"
#include "stm32f1xx_hal.h"

void uart_handle(uint8_t datarx);
void uart_init(UART_HandleTypeDef *huart1);

#endif

