from network_modules import VAN

model = VAN()
model.build((1, 2196, 1958, 3))
model.summary()
