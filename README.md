# Universidad Politécnica Salesiana
## Realizado por : Juan Francisco Quizhpi Fajardo
### Detección de movimiento a través de resta de imágenes, convolución, aplicación de filtros para reducción de ruido y operaciones morfológicas en la mejora de imágenes médicas.
---
+ Función que permita mostrar FPS

![MostrarFPS](https://imgur.com/j29qUme.png)

+ Función que permita detectar movimiento Adaptive Background Substraction

![MostrarFPS](https://imgur.com/S8IJ8np.png)


+ Aplique distintos filtros que permitan mejorar los problemas de iluminación vistos en clase. Para ello debe
probar la ecualización de histograma, el métod CLAHE y uno más que Usted investigue.

### ¿Cómo funciona el filtro gamma?
+  En el filtro gamma, cada pixel que forma parte de la imagen es transforamdo de la siguiente manera

$$
I_{\text{salida}} = I_{\text{entrada}}^{\gamma}
$$


+ I salida es el valor de intensidad del píxel después de aplicar la correción gamma es nuestro resultado
+ I entrada es el valor de la intensidad del píxel original este es normalizado entre 0 y 1
+ "𝛾" Este es el valor de coreción gamma el cual define el valor de ajuste del brillo 

El filtro aplica las siguientes condiciones
+ Si 𝛾<1, el filtro aumenta el brillo de las áreas oscuras sin afectar mucho las áreas brillantes
+ si 𝛾>1, se reduce el brillo de las áreas más claras.

### Comparación del filtro gamma con la Ecualización del Histograma
+ Como ya sabemos las **ecualización del histograma** es una técnica con la que se redistribuye los valores de intensidad de pixeles de la imagen para mejorar su constraste. En comparación al **filtro gamma** la **ecualización de histograma** no debende del parametro 𝛾 sino que se centra en distribuir los valores de intensidad de manera uniforme.

### Comparación con el Filtro CLAHE
+ **CLAHE** es considera una mejora de la ecualización de histograma, **CLAHE** se realiza localmente en la imagen, dividiéndola en pequeñas regiones (tiles) y aplicando la ecualización en cada una de ellas.

### Filtro Gamma
+ Mejora el brillo de manera controlada y personalizada mediante el ajuste de 𝛾, Este filtro es útil si se quiere un ajuste suave sin cambiar significativamente el contraste.

![Filtro gamma](https://imgur.com/ol17x5p.png)


### Método que genere un porcentaje de ruido de sal o pimienta en un video

![SalPimienta](https://imgur.com/8oh4APl.png)

### Aplicar los siguientes filtros: mediana, blur, Gaussiano

![FIltrosMBG](https://imgur.com/89UJdhR.png)
![FIltrosMBG](https://imgur.com/9eHyHFR.png)

### Aplicar al menos 2 algoritmos de detección de bordes y comparar los resultados de usar o no filtros de suavizado.
+ Sobel

![FIltrosSobel](https://imgur.com/jl5CV9P.png)

+ Canny

![FIltrosCanny](https://imgur.com/AlrfOVn.png)


### Seleccionar 3 imágenes médicas a las que se les aplicarán las operaciones morfológicas. Las imágenes deben estar en escala de grises y deben corresponder a radiografías, angiografías, TACs, etc
+ Using morphological transforms to enhance the contrast of medical images1

+ Erosión

![FIltrosErosion](https://imgur.com/l1Tdu0p.png)


+ Dilatación

![FIltrosDilatacion](https://imgur.com/QnYaaLD.png)


+ Top Hat, Black Hat, Imagen Original + (Top Hat – Black Hat)

![FIltrosDilatacion](https://imgur.com/YzPQ4z3.png)


